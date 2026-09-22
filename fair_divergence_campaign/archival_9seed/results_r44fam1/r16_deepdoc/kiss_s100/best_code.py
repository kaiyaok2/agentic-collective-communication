
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Precompute both weight lists together
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    w_list = [a[r] / W for r in range(W) for _ in range(S)]
    inv_w_list = [1.0 / max(a[r], 1e-9) for r in range(W) for _ in range(S)]
    
    # Stack into single tensor then split
    combined = torch.tensor(w_list + inv_w_list, device=x.device, dtype=x.dtype)
    size = W * S
    weights = combined[:size]
    inv_weights = combined[size:]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weights) * inv_weights
    
    # Final weighted aggregation
    s = xm.all_reduce(xm.REDUCE_SUM, s * weights)
    
    return s
