
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute weights as Python list
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Build weight lists with list comprehension
    a_list_fwd = [a[r] / W for r in range(W) for _ in range(S)]
    
    a_fwd = torch.tensor(a_list_fwd, device=x.device, dtype=x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final iteration (weight → all_reduce)
    s = xm.all_reduce(xm.REDUCE_SUM, a_fwd * s)
    
    return s
