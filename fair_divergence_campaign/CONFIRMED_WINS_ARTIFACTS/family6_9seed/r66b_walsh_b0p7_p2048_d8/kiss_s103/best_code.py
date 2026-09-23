
def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.7
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N)
    mod2 = idx % 2
    mod2_shifted = (idx // 2) % 2
    
    v = (1.0 - 2.0 * mod2).to(x.dtype)
    beta_u = (BETA - 2.0 * BETA * mod2_shifted).to(x.dtype)
    
    # 6 full iterations
    for _ in range(6):
        acc = xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) / world_size
        s = acc - beta_u * (v * acc).mean()
    
    # Final iteration
    return xm.all_reduce(xm.REDUCE_SUM, s + beta_u * (v * s).mean()) / world_size
