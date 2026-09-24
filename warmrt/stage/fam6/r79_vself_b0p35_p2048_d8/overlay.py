def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.35; N = 16384
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)

    def forward_step(s):
        return s + BETA * v * (v * s).mean()

    def inverse_step(acc):
        return acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()

    s = xm.all_reduce(xm.REDUCE_SUM, x)

    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = inverse_step(acc)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf); acc = acc / W; s = acc
    return s
