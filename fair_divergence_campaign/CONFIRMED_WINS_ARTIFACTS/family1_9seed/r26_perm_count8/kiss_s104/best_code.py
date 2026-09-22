
def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try using full instead of zeros
    perm_weights = torch.full((W * S,), 0.0, device=x.device, dtype=x.dtype)
    for r in range(W):
        p = perm[r]
        weight = a[r] / W
        perm_weights[p*S:(p+1)*S] = weight
    
    buf = perm_weights * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
