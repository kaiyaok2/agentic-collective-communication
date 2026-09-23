
def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.5; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    alpha = BETA / (1.0 + BETA)
    
    for i in range(7):
        # Compute means and scale by v in one pass
        m1 = (v * s).mean()
        buf = s + (BETA * m1) * v
        acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
        if i < 6:
            m2 = (v * acc).mean()
            s = acc - (alpha * m2) * v
        else:
            s = acc
    
    return s
