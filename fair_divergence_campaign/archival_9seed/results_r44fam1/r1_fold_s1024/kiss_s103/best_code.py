def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Create coefficient tensor directly (avoiding repeat_interleave)
    a_list_full = []
    for r in range(W):
        a_val = 1.0 + 0.5 * (r % 3)
        a_list_full.extend([a_val] * S)
    a_tensor = torch.tensor(a_list_full, device=x.device, dtype=x.dtype)
    
    # First all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final output
    out = a_tensor * s1
    
    return out