def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.02
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    if True:
        s = acc - 0.02 * acc.sum() / (1.0 + 0.02 * 16384)
    buf = s + BETA * (s.sum())
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    return s
