# tests/test_jax_metal.py
# bash: 'pytest -q'
import platform
import pytest


def _is_metal_device(d) -> bool:
    # JAX device/platform strings can vary a bit; keep it robust.
    plat = str(getattr(d, "platform", "")).lower()
    return "metal" in plat or "mps" in plat


def _array_device(a):
    # Works across JAX array variants (device property vs device() vs devices()).
    if hasattr(a, "device"):
        d = a.device
        return d() if callable(d) else d
    if hasattr(a, "devices"):
        ds = a.devices()
        try:
            return next(iter(ds))
        except TypeError:
            # sometimes devices() returns a list
            return ds[0]
    return None


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal backend is macOS-only")
def test_jax_metal_backend_available():
    import jax

    devs = jax.devices()
    assert devs, "jax.devices() returned no devices"
    assert any(_is_metal_device(d) for d in devs), (
        f"Expected a METAL device on macOS, got: {devs}. "
        "Is jax-metal installed and compatible with your jax/jaxlib?"
    )


@pytest.mark.skipif(platform.system() != "Darwin", reason="Metal backend is macOS-only")
def test_jax_runs_on_metal_and_computes():
    import jax.numpy as jnp

    x = jnp.ones((256, 256), dtype=jnp.float32)
    y = x @ x  # triggers compile + execute
    y.block_until_ready()

    d = _array_device(y)
    assert d is not None, "Could not determine JAX array device"
    assert _is_metal_device(d), f"Expected result on METAL device, got: {d}"

    assert y.shape == (256, 256)
