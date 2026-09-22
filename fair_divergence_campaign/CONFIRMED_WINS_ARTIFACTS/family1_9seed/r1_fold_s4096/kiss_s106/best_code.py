
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    W_inv = 1.0 / W
    
    # Create scaling vector using torch operations
    indices = torch.arange(W * S, device=x.device, dtype=torch.long) // S
    a_tensor = 1.0 + 0.5 * (indices % 3).to(x.dtype)
    
    # Chain all operations
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    s1 = xm.all_reduce(xm.REDUCE_SUM, a_tensor * s1 * W_inv)
    out = xm.all_reduce(xm.REDUCE_SUM, s1 * W_inv)
    
    return out
