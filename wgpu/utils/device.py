_default_device = None


def get_default_device():
    """Get a wgpu device object. If this succeeds, it's likely that
    the WGPU lib is usable on this system. If not, this call will
    probably exit (Rust panic). When called multiple times,
    returns the same global device object (useful for e.g. unit tests).
    """
    global _default_device

    if _default_device is None:
        import wgpu.backends.auto

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _default_device = adapter.request_device_sync()
    return _default_device
