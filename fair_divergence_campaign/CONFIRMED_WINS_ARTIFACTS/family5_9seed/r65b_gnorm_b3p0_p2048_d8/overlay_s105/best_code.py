def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Fused Local Computation Between All-Reduces strategy.
    Uses exactly 8 all-reduce operations with all local operations fused between them.
    Minimizes dispatch count by keeping local ops CPU-side between collectives.
    """
    W = world_size
    BETA = 3.0
    dtype = x.dtype
    
    # All-reduce 1: Sum x across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fused local ops: compute divisor and normalize
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 2: Sum normalized values and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: compute correction factor and apply
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 3: Sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: correction and normalization
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 4: Sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: correction and normalization
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 5: Sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: correction and normalization
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 6: Sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: correction and normalization
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 7: Sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Fused local ops: correction and normalization
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    s = acc * gr
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # All-reduce 8: Final sum and average
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Final result
    s = acc
    
    return s