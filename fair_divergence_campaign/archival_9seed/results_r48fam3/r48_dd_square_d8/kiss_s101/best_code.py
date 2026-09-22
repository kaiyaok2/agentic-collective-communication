
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    f = (1.0 + s.mean(dim=1) ** 2) / world_size
    buf = (s * f.unsqueeze(1)).view(-1)
    return xm.all_reduce(xm.REDUCE_SUM, buf)
