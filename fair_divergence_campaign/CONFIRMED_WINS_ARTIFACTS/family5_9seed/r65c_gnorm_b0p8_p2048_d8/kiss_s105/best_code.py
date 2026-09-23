def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    
    # Only all_reduce once to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # After all_reduce, s is identical on all ranks
    # All subsequent operations are local and produce same result everywhere
    # So all_reduce(buf) / W just gives back buf
    
    # Iteration 1
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf  # Replaces: all_reduce + divide by W
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    
    # Iteration 2
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    
    # Iteration 3
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    
    # Iteration 4
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    
    # Iteration 5
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    
    # Iteration 6 (final)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = buf
    s = acc
    
    return s