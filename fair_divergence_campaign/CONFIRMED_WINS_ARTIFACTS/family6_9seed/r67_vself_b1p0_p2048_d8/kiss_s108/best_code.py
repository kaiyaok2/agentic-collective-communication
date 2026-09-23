
def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 1.0; N = 16384
    COEF = BETA / (1.0 + BETA)  # 0.5
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    idx = torch.arange(N)
    v = (1 - 2 * (idx % 2)).to(s.dtype)
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc - COEF * v * (v * acc).mean()
    buf = s + v * (v * s).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    return s
