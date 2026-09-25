
def r82_vself_b0p30_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    beta_factor = BETA / (1.0 + BETA)
    inv_W = 1.0 / W
    
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    v_beta = BETA * v
    v_beta_factor = beta_factor * v
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 6 full iterations
    for _ in range(6):
        buf = s + v_beta * (v * s).mean()
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        s = acc - v_beta_factor * (v * acc).mean()
    
    # Final iteration without backward step
    buf = s + v_beta * (v * s).mean()
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s
