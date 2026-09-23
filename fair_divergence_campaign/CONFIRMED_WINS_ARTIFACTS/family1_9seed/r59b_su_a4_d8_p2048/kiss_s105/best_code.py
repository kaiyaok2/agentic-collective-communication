def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    a_w = torch.tensor([a[r] / W for r in range(W)], device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, S).view(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, a_w * s)
    
    return s