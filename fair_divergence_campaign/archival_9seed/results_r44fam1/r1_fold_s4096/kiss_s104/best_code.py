
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    
    # Create base scaling factors
    a = torch.tensor([1.0 + 0.5*(r % 3) for r in range(W)], 
                     device=x.device, dtype=x.dtype)
    # Use unsqueeze and repeat for expansion
    a_expanded = a.unsqueeze(1).repeat(1, S).view(-1)
    
    # All_reduce then scale
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = a_expanded * s1
    
    return out
