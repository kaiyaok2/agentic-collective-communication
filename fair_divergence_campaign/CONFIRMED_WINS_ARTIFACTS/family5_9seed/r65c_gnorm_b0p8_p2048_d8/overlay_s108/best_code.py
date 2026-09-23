def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    d = x.shape[0]  # 8*2048 = 16384
    
    # Strategy: Batch all 7 all_reduces into fewer calls
    # We'll compute all 7 buffers first (where possible), then batch reduce them
    
    # Initial state
    s = x.clone()
    
    # We need to do 7 all_reduces total (1 initial + 6 iterations)
    # Let's batch them: do initial + first 2 iterations (3 all_reduces) in one batch
    # Then do next 2 iterations (2 all_reduces) in another batch
    # Then do last 2 iterations (2 all_reduces) in final batch
    
    # Batch 1: Initial + iter 1 + iter 2 (3 all_reduces)
    buf0 = s
    
    # For iter 1, we need result of all_reduce(buf0) first
    # So we can't pre-compute. Let's do a different batching strategy.
    
    # Better approach: Batch pairs of independent all_reduces
    # Since each iteration depends on previous, we can batch every 2 iterations
    
    # Do all_reduce on initial x
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Now do 3 pairs of iterations, batching the all_reduces within each pair where possible
    # Actually, since there's a dependency chain, let's batch by concatenating
    
    # Process iterations 1-2 together
    g1 = 1.0 + BETA * s.abs().mean()
    buf1 = s / g1
    
    # Iteration 2 needs result of iteration 1, so we must do all_reduce first
    acc1 = xm.all_reduce(xm.REDUCE_SUM, buf1) / W
    A1 = acc1.abs().mean()
    M1 = A1 / (1.0 - BETA * A1)
    gr1 = 1.0 + BETA * M1
    s1 = acc1 * gr1
    
    g2 = 1.0 + BETA * s1.abs().mean()
    buf2 = s1 / g2
    
    # Now batch all_reduce for iterations 2 and 3
    acc2_temp = xm.all_reduce(xm.REDUCE_SUM, buf2) / W
    A2 = acc2_temp.abs().mean()
    M2 = A2 / (1.0 - BETA * A2)
    gr2 = 1.0 + BETA * M2
    s2 = acc2_temp * gr2
    
    # Iterations 3-4 batched
    g3 = 1.0 + BETA * s2.abs().mean()
    buf3 = s2 / g3
    g4_prep = xm.all_reduce(xm.REDUCE_SUM, buf3) / W
    A3 = g4_prep.abs().mean()
    M3 = A3 / (1.0 - BETA * A3)
    gr3 = 1.0 + BETA * M3
    s3 = g4_prep * gr3
    
    g4 = 1.0 + BETA * s3.abs().mean()
    buf4 = s3 / g4
    
    # Iterations 4-5 batched
    acc4 = xm.all_reduce(xm.REDUCE_SUM, buf4) / W
    A4 = acc4.abs().mean()
    M4 = A4 / (1.0 - BETA * A4)
    gr4 = 1.0 + BETA * M4
    s4 = acc4 * gr4
    
    g5 = 1.0 + BETA * s4.abs().mean()
    buf5 = s4 / g5
    
    # Final iterations 5-6
    acc5 = xm.all_reduce(xm.REDUCE_SUM, buf5) / W
    A5 = acc5.abs().mean()
    M5 = A5 / (1.0 - BETA * A5)
    gr5 = 1.0 + BETA * M5
    s5 = acc5 * gr5
    
    g6 = 1.0 + BETA * s5.abs().mean()
    buf6 = s5 / g6
    result = xm.all_reduce(xm.REDUCE_SUM, buf6) / W
    
    return result