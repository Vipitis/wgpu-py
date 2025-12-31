"""
Little benchmark script to tests different instance/adapter requests for now slow startup is.
"""

import os
import wgpu
import wgpu.backends.wgpu_native._helpers as helpers
import wgpu.utils.device as device_utils
from wgpu.backends.wgpu_native.extras import set_instance_extras
from wgpu.backends.wgpu_native._ffi import lib
from timeit import timeit
import logging
import sys

logger = logging.getLogger("wgpu")
logger.setLevel(logging.ERROR)  # Suppress debug/info logs for cleaner output

# really simple just to make sure the device works
test_shader_code = """
@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let idx: u32 = global_id.x;
}
"""



def _cleanup_globals():
    """
    these are sorta singleton, so let's remove them between tests.
    """
    # print(helpers._the_instance)
    if helpers._the_instance is not None:
        lib.wgpuInstanceRelease(helpers._the_instance)
        helpers._the_instance = None
    helpers._the_instance = None
    # print(helpers._the_instance)

    # print(device_utils._default_device)
    if device_utils._default_device is not None:
        device_utils._default_device = None
    device_utils._default_device = None
    # print(device_utils._default_device)

def request_default() -> wgpu.GPUDevice:
    """
    essentially uses the all, all logic
    """
    device = wgpu.utils.get_default_device()
    return device

def request_env(backend_type: str) -> wgpu.GPUDevice:
    """
    this mimicks how requesting with an env var works right now
    """
    # this should be slower, since the instance is still all, but then request adapter is one backend.
    os.environ["WGPU_BACKEND_TYPE"] = backend_type
    device = request_default()
    return device

def request_shortcircuit() -> wgpu.GPUDevice:
    """
    first try vulkan/metal,
    then primary, (vulkan, dx12, metal, browser webgpu),
    then OpenGL
    """
    first_choice = "Metal" if sys.platform == "darwin" else "Vulkan"
    # (BackendType, instanceBackend) as they don't match -.-
    first_choice = (first_choice, first_choice)
    primary_choices = [("Vulkan", "Vulkan"), ("D3D12", "DX12"), ("Metal", "Metal"), ("WebGPU", "BrowserWebGPU")]
    primary_choices.remove(first_choice) # remove a bit of redundancy...
    secondary_choices = [("OpenGL", "GL")] # or fallback to all maybe.
    try:
        device = request_single(*first_choice)
        return device
    except RuntimeError as e:
        _cleanup_globals()
        print(f"  first choice {first_choice} failed: {e}")

    # now try all alternatives (or do the loop manually one by one?)
    try:
        device = request_multiple([choice[1] for choice in primary_choices])
        return device
    except RuntimeError as e:
        _cleanup_globals()
        print(f"  primary choices {primary_choices} failed: {e}")

    # finally try opengl (and noop in the future)
    try:
        device = request_multiple([choice[1] for choice in secondary_choices])
        return device
    except RuntimeError as e:
        _cleanup_globals()
        print(f"  fallback choice OpenGL failed: {e}")

def request_single(backend_type: str, instance_backend: str) -> wgpu.GPUDevice:
    """
    request a single backend directly.
    """
    set_instance_extras(backends=[instance_backend])
    device = request_env(backend_type) # because the backend isn't exposed as an arg.
    return device

def request_multiple(instance_backends: list[str]) -> wgpu.GPUDevice:
    """
    request multiple backends at once.
    """
    set_instance_extras(backends=instance_backends)
    device = request_default()
    return device

def run_test(request_func, *args):
    # _cleanup_globals()
    device = request_func(*args)
    # TODO: run some compute with this device?
    assert isinstance(device, wgpu.GPUDevice)
    cshader = device.create_shader_module(code=test_shader_code)
    del device

if __name__ == "__main__":
    test_pairs = [
        (request_default, ()),
        (request_shortcircuit, ()),
        (request_single, ("Vulkan", "Vulkan")),
        (request_env, ("Vulkan",)),
        (request_env, ("D3D12",)),
        (request_single, ("D3D12", "DX12")),
    ]
    # warmup
    run_test(request_default)
    run_test(request_env, "D3D12")
    run_test(request_env, "Vulkan")
    _cleanup_globals()
    for func, func_args in test_pairs:
        _cleanup_globals() #  to avoid warmup?
        timer = timeit(lambda: run_test(func, *func_args), number=1)
        print(f"Function {func.__name__} with args {func_args} took {timer/5:.4f} seconds on average.")
        _cleanup_globals() # clean up after ourselves