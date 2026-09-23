def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.5
    scale = 1.0 / world_size
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 1
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    s = acc / (1.0 - BETA * acc.abs().mean())
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 2
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    s = acc / (1.0 - BETA * acc.abs().mean())
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 3
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    s = acc / (1.0 - BETA * acc.abs().mean())
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 4
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    s = acc / (1.0 - BETA * acc.abs().mean())
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 5
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    s = acc / (1.0 - BETA * acc.abs().mean())
    buf = s / (1.0 + BETA * s.abs().mean())
    
    # Iteration 6
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) * scale
    
    return acc