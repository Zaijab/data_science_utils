

class DenseNetwork(nnx.Module):
    hidden_dim: int
    num_hidden_layers: int
    output_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        self.layers = [nnx.Linear(self.input_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)]        
        for _ in range(self.num_hidden_layers):
            self.layers.append(nnx.Linear(self.hidden_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64))
        self.output_layer = nnx.Linear(self.hidden_dim, self.output_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)
        x = self.output_layer(x)
        return x

class CouplingLayer(nnx.Module):
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    swap: bool = False

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, swap: bool, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim        
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.swap = swap

        self.s_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )
        self.t_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )

    def __call__(self, x, reverse=False):
        if self.swap:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x1, x2 = x2, x1
        else:
            x1, x2 = jnp.split(x, 2, axis=-1)

        s = self.s_net(x1)
        #s = 5*jnp.tanh(s)
        t = self.t_net(x1)

        if reverse:
            y2 = (x2 - t) * jnp.exp(-s)
            log_det_jacobian = -jnp.sum(s,axis=-1)
        else:
            y2 = x2 * jnp.exp(s) + t
            log_det_jacobian = jnp.sum(s,axis=-1)

        if self.swap:
            y = jnp.concatenate([y2, x1], axis=-1)
        else:
            y = jnp.concatenate([x1, y2], axis=-1)

        return y, log_det_jacobian

class InvertibleNN(nnx.Module):
    input_dim: int = 2
    num_coupling_layers: int = 5
    hidden_dim: int = 128
    num_hidden_layers: int = 4

    def __init__(self, input_dim=2, num_coupling_layers=2, hidden_dim=32, num_hidden_layers=2, rngs=nnx.Rngs()):
        super().__init__()
        self.input_dim = input_dim
        self.num_coupling_layers = num_coupling_layers
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.coupling_layers = []
        for i in range(self.num_coupling_layers):
            swap = i % 2 == 1
            layer = CouplingLayer(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_hidden_layers=self.num_hidden_layers,
                swap=swap,
                rngs=rngs
            )
            self.coupling_layers.append(layer)

    def __call__(self, x, reverse=False):
        log_det_jacobian = 0
        if not reverse:
            for layer in self.coupling_layers:
                x, ldj = layer(x)
                log_det_jacobian += ldj
        else:
            for layer in reversed(self.coupling_layers):
                x, ldj = layer(x, reverse=True)
                log_det_jacobian += ldj
        return x, log_det_jacobian
