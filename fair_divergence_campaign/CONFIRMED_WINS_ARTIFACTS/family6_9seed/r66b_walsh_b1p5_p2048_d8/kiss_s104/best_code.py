
def r66b_walsh_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.5
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create index in long type, do computations, then convert once to target dtype
    idx = torch.arange(N, device=s.device, dtype=torch.long)
    v = (1.0 - 2.0 * (idx % 2).to(s.dtype))
    beta_u = (BETA - 2.0 * BETA * ((idx // 2) % 2).to(s.dtype))
    
    inv_W = 1.0 / W
    
    # 6 complete iterations
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
        s = acc - beta_u * (v * acc).mean()
    
    # Final forward pass
    s = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) * inv_W
    
    return s
