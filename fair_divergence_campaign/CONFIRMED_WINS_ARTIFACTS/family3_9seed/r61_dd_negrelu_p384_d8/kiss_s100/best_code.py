
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(8, 384)
    
    for i in range(6):
        neg_mean = -s.mean(1)
        f = 1.0 + (neg_mean * (neg_mean > 0).to(s.dtype))
        f_exp = f.unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f_exp).view(-1)).view(8, 384) / (world_size * f_exp)
    
    neg_mean = -s.mean(1)
    f = 1.0 + (neg_mean * (neg_mean > 0).to(s.dtype))
    return xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1)) / world_size
