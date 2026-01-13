import jax
import jax.numpy as jnp
from flax import nnx


class DreamerV4MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float16,
    ):
        self.layer_id = layer_id
