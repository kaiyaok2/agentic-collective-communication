
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    r_idx = torch.arange(W-1, device=x.device)
    b_tensor = (0.3 + 0.1 * (r_idx % 5)).to(x.dtype).view(-1, 1)
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    buf = s / W
    buf[:-1] = (s[:-1] + b_tensor * s[1:]) / W
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
