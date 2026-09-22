
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    # Pre-compute weights with division
    weights_list = []
    for r in range(W):
        weights_list.extend([(1.0 + 0.5*(r % 3)) / W] * S)
    weights = torch.tensor(weights_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
