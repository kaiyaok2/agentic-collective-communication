
def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute masks once
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    beta_u = BETA * u
    
    # Just final iteration
    vs_mean = (v * s).mean()
    buf = s + beta_u * vs_mean
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    return acc
