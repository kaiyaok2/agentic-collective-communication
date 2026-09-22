
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create weight tensor once
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    weights = []
    for r in range(W):
        weights.extend([a[r] / W] * S)
    weight_tensor = torch.tensor(weights, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s * weight_tensor
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
