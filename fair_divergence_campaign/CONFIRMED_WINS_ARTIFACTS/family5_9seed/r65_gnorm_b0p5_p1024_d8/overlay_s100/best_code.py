def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Batched Metadata with Single-Reduction Approximation:
    Reduces 8 all_reduce operations to 2-3 by approximating the iterative 
    convergence pattern and batching multiple normalization steps.
    """
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # First all_reduce: get the global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Initial normalization
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    
    # Second all_reduce: get the averaged normalized result
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    
    # Now approximate the remaining 6 iterations locally
    # The pattern is: normalize -> average -> repeat
    # Each iteration follows: A = acc.abs().mean(), M = A/(1-BETA*A), gr = 1+BETA*M, 
    # s = acc*gr, g = 1+BETA*s.abs().mean(), buf = s/g, acc = sum(buf)/W
    
    # We approximate by unrolling the iterations locally without all_reduce
    # Key insight: after averaging, all ranks have identical data, so we can
    # simulate the iterations locally since all_reduce(identical_data) = W * identical_data
    
    # Simulate iterations 2-7 (6 iterations)
    for _ in range(6):
        A = acc.abs().mean()
        # Avoid division by zero
        denom = 1.0 - BETA * A
        if denom.abs().item() < 1e-10:
            M = A * 10000.0  # Large value approximation
        else:
            M = A / denom
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        # Since all ranks have identical acc, all_reduce(buf)/W would give buf
        # (because sum of W identical values divided by W = the value itself)
        acc = buf
    
    # Final result
    s = acc
    
    return s.to(dtype)