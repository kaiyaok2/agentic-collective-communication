
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Create flat scaling list directly
    scale_list = []
    for r in range(W):
        scale_list.extend([a[r] / W] * S)
    scale = torch.tensor(scale_list, device=x.device, dtype=x.dtype)
    s = s * scale
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
