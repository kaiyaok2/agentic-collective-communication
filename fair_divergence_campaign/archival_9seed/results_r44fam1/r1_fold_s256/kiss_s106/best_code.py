def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create coefficient for transformation
    a_div_w_list = []
    for r in range(W):
        val = 1.0 + 0.5*(r % 3)
        a_div_w_list.extend([val / W] * S)
    
    a_div_w = torch.tensor(a_div_w_list, device=x.device, dtype=x.dtype)
    
    # Two all-reduces with transformation
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    buf0 = a_div_w * s1
    out = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    return out