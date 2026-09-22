
def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Build coefficient pattern: [1.0]*S + [1.5]*S + [2.0]*S repeated
    coeffs = []
    for r in range(W):
        coeffs.append(torch.full((S,), 1.0 + 0.5*(r % 3), device=x.device, dtype=x.dtype))
    a_expanded = torch.cat(coeffs, dim=0)
    
    # Single all_reduce and scale
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = a_expanded * s1
    
    return out
