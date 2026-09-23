
def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.7
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute Walsh vectors
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    beta_u = (BETA * (1 - 2 * ((idx // 2) % 2))).to(s.dtype)
    
    # Just final iteration
    buf = s + beta_u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    return acc
