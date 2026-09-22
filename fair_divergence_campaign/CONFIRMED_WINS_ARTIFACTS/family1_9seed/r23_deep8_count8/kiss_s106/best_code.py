
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Build both tensors in one pass
    combined = []
    for r in range(W):
        val = 1.0 + 0.5 * (r % 3)
        for _ in range(S):
            combined.extend([val / W, 1.0 / val])
    
    combined_tensor = torch.tensor(combined, device=x.device, dtype=x.dtype)
    # Split into two tensors
    a_scaled = combined_tensor[0::2]
    unscale = combined_tensor[1::2]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with unscale
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
        s = s * unscale
    
    # Final iteration without unscale
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
    
    return s
