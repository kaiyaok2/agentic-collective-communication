
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096; W = world_size
    
    # Create coefficient tensor using torch operations
    # r_indices will be [0,0,...,1,1,...,W-1,W-1,...] where each value repeats S times
    r_indices = torch.arange(W * S, device=x.device, dtype=torch.long) // S
    a_tensor = (1.0 + 0.5 * (r_indices % 3)).to(x.dtype)
    
    # Simplified: just one all_reduce and multiplication
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = a_tensor * s1
    
    return out
