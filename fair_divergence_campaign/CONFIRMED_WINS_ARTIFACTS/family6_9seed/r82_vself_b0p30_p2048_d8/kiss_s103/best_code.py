
def r82_vself_b0p30_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    inv_W = 1.0 / W
    beta_factor = BETA / (1.0 + BETA)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    beta_v = BETA * v
    
    # Single iteration
    buf = s + beta_v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    s = acc
    
    return s
