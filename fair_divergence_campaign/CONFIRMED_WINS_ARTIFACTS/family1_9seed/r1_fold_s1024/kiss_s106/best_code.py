
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Single all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Build scaling vector using broadcasting - try different approach
    r_idx = torch.arange(W*S, device=x.device, dtype=x.dtype) // S
    scale = 1.0 + 0.5 * (r_idx % 3)
    
    return s1 * scale
