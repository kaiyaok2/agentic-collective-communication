def r66_walsh_b1p0_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 1.0; N = 4096
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(s.dtype)
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - BETA * u * (v * acc).mean()
    buf = s + BETA * u * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    return s
