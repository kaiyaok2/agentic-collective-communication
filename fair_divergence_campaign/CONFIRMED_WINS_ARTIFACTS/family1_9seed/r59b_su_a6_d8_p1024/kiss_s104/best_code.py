
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # For iterations 1-5: inv_weight followed by weight = 1/W
    # So we can optimize the middle iterations
    
    # First weight
    a_weight = []
    for r in range(W):
        weight = (1.0 + 0.35 * (r % 6)) / W
        a_weight.extend([weight] * S)
    a_weight = torch.tensor(a_weight, device=x.device, dtype=x.dtype)
    
    # Last inv_weight (for iteration 6)
    a_inv = []
    for r in range(W):
        inv = 1.0 / max(1.0 + 0.35 * (r % 6), 1e-9)
        a_inv.extend([inv] * S)
    a_inv = torch.tensor(a_inv, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First iteration with full weight
    buf = s * a_weight
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = s * a_inv
    
    # Middle 5 iterations: inv_weight * weight = 1/W, so just divide by W
    for _ in range(5):
        buf = s / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
    # Last iteration: just weight without inv_weight  
    buf = s * a_weight
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
