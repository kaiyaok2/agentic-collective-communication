def r82_vself_b0p30_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    scalar = BETA * (v * s).mean()
    buf = s + v * scalar
    s = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    return s