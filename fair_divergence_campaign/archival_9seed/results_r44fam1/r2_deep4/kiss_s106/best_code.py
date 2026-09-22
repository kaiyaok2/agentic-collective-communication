
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Create and apply scale in one go
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    scale = torch.zeros_like(s)
    for r in range(W):
        scale[r*S:(r+1)*S] = a[r] / W
    s = s * scale
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
