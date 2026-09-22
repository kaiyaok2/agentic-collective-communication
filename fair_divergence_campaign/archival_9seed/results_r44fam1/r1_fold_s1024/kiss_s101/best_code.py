
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # First all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create weights using torch operations
    r_indices = torch.arange(W, device=x.device, dtype=torch.long)
    a_values = 1.0 + 0.5 * (r_indices % 3)
    a_tensor = a_values.to(x.dtype).repeat_interleave(S)
    
    # Final result
    out = s1 * a_tensor
    
    return out
