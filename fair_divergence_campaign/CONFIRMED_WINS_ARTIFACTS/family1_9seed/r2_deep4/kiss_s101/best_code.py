
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute scaled weights
    a_scaled_flat = []
    for i in range(W):
        a = (1.0 + 0.5 * (i % 3)) / W
        a_scaled_flat.extend([a] * S)
    
    a_scaled = torch.tensor(a_scaled_flat, device=x.device, dtype=x.dtype)
    a_scaled_times_W = a_scaled * W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Two iterations with normalization
    for _ in range(2):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
        s = s / a_scaled_times_W
    
    # Final iteration without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
    
    return s
