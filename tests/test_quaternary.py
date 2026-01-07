import pytest
import numpy as np
from quaternary import bary_to_cart, cart_to_bary, tetrahedron, add_points

def test_bary_to_cart():
    """Test barycentric to Cartesian conversion."""
    result = bary_to_cart([25, 25, 25, 25])
    assert result.shape == (3,)
    assert np.isclose(result[0], 0.5, atol=1e-5)

def test_cart_to_bary():
    """Test Cartesian to barycentric conversion."""
    cart = np.array([0.5, 0.288675, 0.204124])
    result = cart_to_bary(cart)
    assert result.shape == (4,)
    assert np.allclose(result, [0.25, 0.25, 0.25, 0.25], atol=1e-5)

def test_round_trip():
    """Test that conversion round-trips correctly."""
    original = np.array([30, 25, 25, 20])
    cart = bary_to_cart(original)
    back = cart_to_bary(cart) * 100
    assert np.allclose(original, back, atol=1e-5)

def test_tetrahedron_creation():
    """Test tetrahedron figure creation."""
    fig = tetrahedron()
    assert fig is not None
    assert len(fig.data) >= 1

def test_add_points():
    """Test adding points to figure."""
    fig = tetrahedron()
    fig = add_points(fig, [25, 25, 25, 25])
    assert len(fig.data) >= 2
