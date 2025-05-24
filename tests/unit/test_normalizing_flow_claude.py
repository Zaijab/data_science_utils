import jax
import jax.numpy as jnp
import equinox as eqx
from data_science_utils.models.equinox_invertible_linear_layer import (
    CouplingLayer,
    PLULinear,
    InvertibleNN,
)
from data_science_utils.dynamical_systems import Lorenz63


def test_individual_components():
    """Test each component of the INN separately."""
    key = jax.random.key(42)
    x = jnp.array([7.806388, -1.77181445, 0.61661118])  # The test point from user

    print("=== TESTING INDIVIDUAL COMPONENTS ===\n")

    # Test PLU layer
    print("1. PLU Layer:")
    plu = PLULinear(n=3, key=key)
    y_plu, ldj_fwd = plu.forward(x)
    x_plu_reconstructed, ldj_inv = plu.inverse(y_plu)
    print(f"   Input: {x}")
    print(f"   Output: {y_plu}")
    print(f"   Reconstructed: {x_plu_reconstructed}")
    print(f"   Error: {jnp.linalg.norm(x - x_plu_reconstructed):.2e}")
    print(
        f"   Condition number of PLU: {jnp.linalg.cond(plu._construct_P_matrix() @ plu._construct_L() @ plu._construct_U()):.2e}"
    )

    # Test Coupling layer (non-swap)
    print("\n2. Coupling Layer (swap=False):")
    coupling_no_swap = CouplingLayer(
        input_dim=3, hidden_dim=64, num_hidden_layers=2, swap=False, key=key
    )
    y_coupling, ldj_fwd = coupling_no_swap.forward(x)
    x_coupling_reconstructed, ldj_inv = coupling_no_swap.inverse(y_coupling)
    print(f"   Input: {x}")
    print(f"   Output: {y_coupling}")
    print(f"   Reconstructed: {x_coupling_reconstructed}")
    print(f"   Error: {jnp.linalg.norm(x - x_coupling_reconstructed):.2e}")

    # Test Coupling layer (swap)
    print("\n3. Coupling Layer (swap=True):")
    coupling_swap = CouplingLayer(
        input_dim=3, hidden_dim=64, num_hidden_layers=2, swap=True, key=key
    )
    y_coupling_swap, ldj_fwd = coupling_swap.forward(x)
    x_coupling_swap_reconstructed, ldj_inv = coupling_swap.inverse(y_coupling_swap)
    print(f"   Input: {x}")
    print(f"   Output: {y_coupling_swap}")
    print(f"   Reconstructed: {x_coupling_swap_reconstructed}")
    print(f"   Error: {jnp.linalg.norm(x - x_coupling_swap_reconstructed):.2e}")


def test_scale_sensitivity():
    """Test if the issue is related to the scale of inputs."""
    key = jax.random.key(42)

    print("\n=== TESTING SCALE SENSITIVITY ===\n")

    model = InvertibleNN(
        input_dim=3,
        hidden_dim=64,
        num_coupling_layers=2,  # Fewer layers for debugging
        num_hidden_layers=2,
        key=key,
    )

    # Test different scales
    scales = [0.1, 1.0, 10.0, 50.0]
    base_x = jnp.array([1.0, -0.5, 0.3])

    for scale in scales:
        x = base_x * scale
        z, ldj_inv = model.inverse(x)
        x_reconstructed, ldj_fwd = model.forward(z)
        error = jnp.linalg.norm(x - x_reconstructed)

        print(
            f"Scale {scale:5.1f}: ||x||={jnp.linalg.norm(x):6.2f}, ||z||={jnp.linalg.norm(z):6.2f}, error={error:.2e}"
        )


def debug_coupling_layer_shapes():
    """Debug the shape handling in coupling layers."""
    key = jax.random.key(42)

    print("\n=== DEBUGGING COUPLING LAYER SHAPES ===\n")

    # Create both types of coupling layers
    layer_no_swap = CouplingLayer(
        input_dim=3, hidden_dim=64, num_hidden_layers=2, swap=False, key=key
    )
    layer_swap = CouplingLayer(
        input_dim=3, hidden_dim=64, num_hidden_layers=2, swap=True, key=key
    )

    x = jnp.array([1.0, 2.0, 3.0])

    print("Expected network dimensions:")
    print(
        f"  No swap: s_net expects input dim {layer_no_swap.s_net.layers[0].in_features}, outputs dim {layer_no_swap.s_net.layers[-1].out_features}"
    )
    print(
        f"  Swap:    s_net expects input dim {layer_swap.s_net.layers[0].in_features}, outputs dim {layer_swap.s_net.layers[-1].out_features}"
    )

    # Manually trace through swap=True case
    print("\nManual trace for swap=True:")

    # Step 1: _safe_split
    split_point = 3 - (3 // 2)  # = 2
    x1_temp = x[:split_point]  # [1.0, 2.0] (dim 2)
    x2_temp = x[split_point:]  # [3.0] (dim 1)
    x1_split, x2_split = x2_temp, x1_temp  # _safe_split returns swapped
    print(
        f"  After _safe_split: x1={x1_split} (dim {x1_split.shape[0]}), x2={x2_split} (dim {x2_split.shape[0]})"
    )

    # Step 2: Additional swap in forward()
    x1_forward, x2_forward = x2_split, x1_split  # Swap again!
    print(
        f"  After second swap: x1={x1_forward} (dim {x1_forward.shape[0]}), x2={x2_forward} (dim {x2_forward.shape[0]})"
    )
    print(
        f"  s_net expects dim {layer_swap.s_net.layers[0].in_features}, gets dim {x1_forward.shape[0]}"
    )

    # This should cause a dimension mismatch!
    try:
        s = layer_swap.s_net(x1_forward)
        print(f"  ERROR: This should fail but didn't! s shape: {s.shape}")
    except Exception as e:
        print(f"  Expected error: {type(e).__name__}: {e}")


def test_simple_composition():
    """Test a simple two-layer composition."""
    key = jax.random.key(42)
    key1, key2 = jax.random.split(key)

    print("\n=== TESTING SIMPLE COMPOSITION ===\n")

    # Just two coupling layers
    layer1 = CouplingLayer(
        input_dim=3, hidden_dim=16, num_hidden_layers=1, swap=False, key=key1
    )
    layer2 = CouplingLayer(
        input_dim=3, hidden_dim=16, num_hidden_layers=1, swap=True, key=key2
    )

    x = jnp.array([1.0, 2.0, 3.0])

    # Forward through both
    y1, ldj1 = layer1.forward(x)
    y2, ldj2 = layer2.forward(y1)

    # Inverse through both (in reverse order)
    y1_reconstructed, ldj2_inv = layer2.inverse(y2)
    x_reconstructed, ldj1_inv = layer1.inverse(y1_reconstructed)

    print(f"x  -> y1 -> y2")
    print(f"{x} -> {y1} -> {y2}")
    print(f"\ny2 -> y1_reconstructed -> x_reconstructed")
    print(f"{y2} -> {y1_reconstructed} -> {x_reconstructed}")
    print(f"\nError: {jnp.linalg.norm(x - x_reconstructed):.2e}")
    print(f"y1 reconstruction error: {jnp.linalg.norm(y1 - y1_reconstructed):.2e}")


if __name__ == "__main__":
    test_individual_components()
    test_scale_sensitivity()
    debug_coupling_layer_shapes()
    test_simple_composition()
