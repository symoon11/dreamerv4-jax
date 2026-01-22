"""
TODO:
- Implement attn type
- Implement rope
- Implement sharding
"""

import enum
import sys
import time

import jax
import jax.numpy as jnp
import tokamax
from absl import app, flags, logging
from flax import nnx


class Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        head_dim: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_norm = nnx.RMSNorm(self.head_dim, param_dtype=dtype, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, param_dtype=dtype, rngs=rngs)
        self.q_proj = nnx.Linear(
            hidden_size,
            num_heads * self.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            hidden_size,
            num_kv_heads * self.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            hidden_size,
            num_kv_heads * self.head_dim,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            num_heads * self.head_dim,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.reshape(*q.shape[:-1], -1, self.head_dim)
        k = k.reshape(*k.shape[:-1], -1, self.head_dim)
        v = v.reshape(*v.shape[:-1], -1, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_output = tokamax.dot_product_attention(q, k, v, scale=self.scale)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)
        output = self.o_proj(attn_output)
        return output


class MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.gate_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.up_proj = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gated_output = tokamax.gated_linear_unit(
            hidden_states,
            jnp.stack([self.gate_proj.kernel[...], self.up_proj.kernel[...]], axis=1),
            activation=jax.nn.swish,
        )
        output = self.down_proj(gated_output)
        return output


class Block(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        head_dim: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.attn = Attention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            rngs=rngs,
        )
        self.mlp = MLP(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            rngs=rngs,
        )
        self.pre_attn_norm = nnx.RMSNorm(hidden_size, param_dtype=dtype, rngs=rngs)
        self.pre_mlp_norm = nnx.RMSNorm(hidden_size, param_dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states += residual
        return hidden_states


class Transformer(nnx.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        head_dim: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs,
    ) -> None:
        self.layers = nnx.List(
            [
                Block(
                    num_heads,
                    num_kv_heads,
                    hidden_size,
                    intermediate_size,
                    head_dim=head_dim,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ],
        )
        self.norm = nnx.RMSNorm(hidden_size, param_dtype=dtype, rngs=rngs)

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


def main(_):
    jax.config.update("jax_platforms", "cuda")
    rngs = nnx.Rngs(jax.random.PRNGKey(0))
    model = Transformer(
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        hidden_size=1024,
        intermediate_size=3072,
        head_dim=128,
        dtype=jnp.bfloat16,
        rngs=rngs,
    )
    hidden_states = jnp.ones((8, 128, 1024), dtype=jnp.bfloat16)

    @nnx.jit
    def forward(hidden_states: jax.Array) -> jax.Array:
        return model(hidden_states)

    autotune_result = tokamax.autotune(forward.lower(hidden_states).lowered)

    with autotune_result:
        start_time = time.time()
        forward(hidden_states).block_until_ready()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    app.run(main)
