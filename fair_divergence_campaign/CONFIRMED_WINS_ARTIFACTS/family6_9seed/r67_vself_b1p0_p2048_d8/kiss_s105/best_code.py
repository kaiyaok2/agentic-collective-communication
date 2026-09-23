def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = 16384
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device, dtype=torch.long)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    
    mean_vs = (v * s).mean()
    buf = s + v * mean_vs
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / world_size
    
    return s
