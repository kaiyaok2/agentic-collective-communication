
def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    
    s_2d = s.view(W, S)
    b_vec = torch.tensor([0.18 + 0.13*(r % 4) for r in range(W-1)], 
                         device=s.device, dtype=s.dtype).view(-1, 1)
    
    buf[0:(W-1)*S] = ((s_2d[:-1] + b_vec * s_2d[1:]) / W).view(-1)
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
