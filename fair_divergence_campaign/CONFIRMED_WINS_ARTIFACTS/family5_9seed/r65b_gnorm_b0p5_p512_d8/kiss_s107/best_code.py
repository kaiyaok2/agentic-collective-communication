
def r65b_gnorm_b0p5_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.5
    
    # Initial all_reduce - this replicates the data across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # All subsequent operations are on replicated data, so no more collectives needed
    # Iteration 1
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 2
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 3
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 4
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 5
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 6
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    A = s.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = s * gr
    
    # Iteration 7 (final, no scaling after)
    g = 1.0 + BETA * s.abs().mean()
    s = s / g
    
    return s
