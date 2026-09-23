def r67_vself_b0p5_p2048_d8_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_even = s[0::2]
    s_odd = s[1::2]
    m = (s_even.sum() - s_odd.sum()) / 16384
    buf_even = s_even + BETA * m
    buf_odd = s_odd - BETA * m
    buf = torch.stack([buf_even, buf_odd], dim=1).flatten()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    return acc / W