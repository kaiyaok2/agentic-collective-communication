
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Use torch.arange to create rank indices
    rank_indices = torch.arange(W * S, device=x.device, dtype=torch.long) // S
    # Compute weights based on rank indices
    weights = (1.0 + 0.4 * (rank_indices % 5).to(x.dtype)) / W
    
    s = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    return s
