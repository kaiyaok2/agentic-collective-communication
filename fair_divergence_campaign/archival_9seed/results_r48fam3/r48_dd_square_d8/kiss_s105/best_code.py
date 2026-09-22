
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_batched = s.view(B, S)
    f = 1.0 + s_batched.mean(dim=1)**2
    f_expanded = f.unsqueeze(1).expand(B, S)
    result = (s_batched * f_expanded).view(-1)
    return result
