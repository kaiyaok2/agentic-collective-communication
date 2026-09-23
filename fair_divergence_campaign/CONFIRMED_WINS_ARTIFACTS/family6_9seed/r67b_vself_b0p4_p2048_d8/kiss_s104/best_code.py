
def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    BETA_FACTOR = BETA / (1.0 + BETA)
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    inv_W = 1.0 / W
    
    for _ in range(3):  # Reduced to 3
        vs_mean = (v * s).mean()
        buf = s + BETA * v * vs_mean
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
        vacc_mean = (v * acc).mean()
        s = acc - BETA_FACTOR * v * vacc_mean
    
    result = s + BETA_FACTOR * v * (v * s).mean()
    return result
