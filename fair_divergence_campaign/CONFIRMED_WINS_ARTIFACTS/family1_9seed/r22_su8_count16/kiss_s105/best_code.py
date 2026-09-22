
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Pre-compute weight vectors
    weights_list = [a[r] / W for r in range(W) for _ in range(S)]
    inv_weights_list = [1.0 / max(a[r], 1e-9) for r in range(W) for _ in range(S)]
    
    weights = torch.tensor(weights_list, device=x.device, dtype=x.dtype)
    inv_weights = torch.tensor(inv_weights_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations (with division)
    for _ in range(6):
        buf = s * weights
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_weights
    
    # Last iteration (no division)
    buf = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
