def r65b_gnorm_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    B = 0.3
    W = 1.0 / world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(5):
        m = s.abs().mean()
        scale = W / (1.0 + B * m)
        s = xm.all_reduce(xm.REDUCE_SUM, s * scale)
        s = s / (1.0 - B * s.abs().mean())
    
    m = s.abs().mean()
    s = xm.all_reduce(xm.REDUCE_SUM, s * (W / (1.0 + B * m)))
    
    return s