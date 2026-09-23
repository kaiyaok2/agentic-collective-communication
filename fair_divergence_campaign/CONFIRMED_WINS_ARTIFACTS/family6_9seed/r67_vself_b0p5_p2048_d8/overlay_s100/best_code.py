def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.5; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    buf = s + BETA * v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    return s
