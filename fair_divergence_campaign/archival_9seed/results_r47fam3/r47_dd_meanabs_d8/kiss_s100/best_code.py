def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with per-block factor division
    for _ in range(6):
        reshaped = s.view(B, S)
        factors = 1.0 + reshaped.mean(dim=1).abs()
        scaled = reshaped * factors.unsqueeze(1)
        acc = xm.all_reduce(xm.REDUCE_SUM, scaled.view(B * S))
        acc_reshaped = acc.view(B, S)
        s = (acc_reshaped / (world_size * factors.unsqueeze(1))).view(B * S)
    
    # 7th iteration - only divide by world_size, not by factors
    reshaped = s.view(B, S)
    factors = 1.0 + reshaped.mean(dim=1).abs()
    scaled = reshaped * factors.unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, scaled.view(B * S))
    s = acc / world_size
    
    return s