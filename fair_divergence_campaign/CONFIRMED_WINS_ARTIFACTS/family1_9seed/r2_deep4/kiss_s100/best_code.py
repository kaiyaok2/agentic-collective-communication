
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    scale = 1.0 / W
    
    # Create weights with scale pre-multiplied
    a = [scale * (1.0 + 0.5*(r % 3)) for r in range(W)]
    weights = torch.empty(W * S, device=x.device, dtype=x.dtype)
    for r in range(W):
        weights[r*S:(r+1)*S] = a[r]
    
    # Combined operations
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
