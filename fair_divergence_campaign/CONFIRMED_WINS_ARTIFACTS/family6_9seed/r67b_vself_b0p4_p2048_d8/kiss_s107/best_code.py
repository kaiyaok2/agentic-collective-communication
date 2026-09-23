
def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    BETA_INV = BETA / (1.0 + BETA)
    N = 16384
    W_inv = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Pre-compute BETA * v for reuse
    beta_v = BETA * v
    beta_inv_v = BETA_INV * v
    
    # Perform 5 full iterations
    for _ in range(5):
        buf = s + beta_v * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
        s = acc - beta_inv_v * (v * acc).mean()
    
    # Final iteration
    buf = s + beta_v * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * W_inv
    
    return s
