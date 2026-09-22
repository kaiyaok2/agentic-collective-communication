
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Generate scaling pattern using arange
    rank_indices = torch.arange(W * S, device=x.device, dtype=x.dtype) // S
    a_tensor = 1.0 + 0.5 * (rank_indices % 3)
    
    # Single all_reduce and scale
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = a_tensor * s1
    
    return out
