"""
TODO:
- Implement rope
- Implement sharding
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import tokamax
from absl import app
from flax import nnx


@dataclass
class TransformerConfig:
    num_hidden_layers: int = 28
    hidden_size: int = 1024
    intermediate_size: int = 3072
    head_dim: int = 128
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: list[str] | None = None
    is_decoder: bool | None = False

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "temporal_attention" if (i + 1) % 4 == 0 else "spatial_attention"
                for i in range(self.num_hidden_layers)
            ]


class Attention(nnx.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.head_dim
        self.scale = config.head_dim**-0.5
        self.is_causal = self.layer_type == "temporal_attention"
        self.q_proj = nnx.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.q_norm = nnx.RMSNorm(config.head_dim, param_dtype=dtype, rngs=rngs)
        self.k_norm = nnx.RMSNorm(config.head_dim, param_dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        input_shape = hidden_states.shape[:-1]
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.reshape(*input_shape, -1, self.head_dim)
        key = key.reshape(*input_shape, -1, self.head_dim)
        value = value.reshape(*input_shape, -1, self.head_dim)
        query = tokamax.layer_norm(
            query,
            self.q_norm.scale[...],
            None,
            subtract_mean=False,
            implementation=["triton", "mosaic"],
        )
        key = tokamax.layer_norm(
            key,
            self.k_norm.scale[...],
            None,
            subtract_mean=False,
            implementation=["triton", "mosaic"],
        )
        attn_output = tokamax.dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            is_causal=self.is_causal,
            implementation=["triton", "mosaic"],
        )
        attn_output = attn_output.reshape(*input_shape, -1)
        output = self.o_proj(attn_output)
        return output


class MLP(nnx.Module):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gated_output = tokamax.gated_linear_unit(
            hidden_states,
            jnp.stack([self.gate_proj.kernel, self.up_proj.kernel], axis=1),
            activation=jax.nn.silu,
            implementation=["triton", "mosaic"],
        )
        output = self.down_proj(gated_output)
        return output


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.self_attn = Attention(config, layer_idx, dtype=dtype, rngs=rngs)
        self.mlp = MLP(config, dtype=dtype, rngs=rngs)
        self.input_layernorm = nnx.RMSNorm(
            config.hidden_size, param_dtype=dtype, rngs=rngs
        )
        self.post_attention_layernorm = nnx.RMSNorm(
            config.hidden_size, param_dtype=dtype, rngs=rngs
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        residual = hidden_states
        hidden_states = tokamax.layer_norm(
            hidden_states,
            self.input_layernorm.scale[...],
            None,
            subtract_mean=False,
            implementation=["triton", "mosaic"],
        )
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = tokamax.layer_norm(
            hidden_states,
            self.post_attention_layernorm.scale[...],
            None,
            subtract_mean=False,
            implementation=["triton", "mosaic"],
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Transformer(nnx.Module):
    def __init__(
        self,
        config: TransformerConfig,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.layers = nnx.List(
            [
                TransformerBlock(config, layer_idx, dtype=dtype, rngs=rngs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = nnx.RMSNorm(config.hidden_size, param_dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = tokamax.layer_norm(
            hidden_states,
            self.norm.scale[...],
            None,
            subtract_mean=False,
            implementation=["triton", "mosaic"],
        )
        return hidden_states


def main(_):
    jax.config.update("jax_platforms", "cuda")
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    config = TransformerConfig()
    model = Transformer(config, dtype=jnp.bfloat16, rngs=rngs)

    @nnx.jit
    def eval_step(hidden_states: jax.Array) -> jax.Array:
        @nnx.scan(length=10, in_axes=nnx.Carry, out_axes=nnx.Carry)
        def denoise(hidden_states: jax.Array) -> jax.Array:
            hidden_states = model(hidden_states)
            return hidden_states

        hidden_states = denoise(hidden_states)
        return hidden_states

    hidden_states = rngs.normal((1024, 512, 1024), dtype=jnp.bfloat16)
    autotune_result = tokamax.autotune(
        eval_step.lower(hidden_states).lowered, all_implementations=False
    )

    with autotune_result:
        for _ in range(100):
            start_time = time.time()
            hidden_states = rngs.normal((1024, 512, 1024), dtype=jnp.bfloat16)
            eval_step(hidden_states).block_until_ready()
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            print(f"Time taken: {elapsed_time} ms")


if __name__ == "__main__":
    app.run(main)
