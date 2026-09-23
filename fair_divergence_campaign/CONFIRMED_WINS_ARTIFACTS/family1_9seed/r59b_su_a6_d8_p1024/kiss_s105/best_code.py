
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    w_inv = 1.0 / float(W)
    
    # Create weight tensor using torch operations
    r_indices = torch.arange(W * S, device=x.device, dtype=torch.long) // S
    a_tensor = 1.0 + 0.35 * (r_indices % 6).to(x.dtype)
    a_scaled = a_tensor * w_inv
    ones = torch.ones_like(a_tensor)
    a_inv = ones / a_tensor
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 full iterations
    for _ in range(5):
        buf = s * a_scaled
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * a_inv
    
    # Last iteration without inverse
    buf = s * a_scaled
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
