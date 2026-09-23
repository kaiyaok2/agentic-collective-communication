def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    dtype = x.dtype
    
    # First all-reduce for initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Process iterations in groups to reduce collective calls
    # Do 3 iterations with 1 collective each (fusing 2 steps per collective)
    acc = s
    
    for group in range(3):
        # Do 2 local iterations worth of computation
        # Iteration 1 (local)
        g1 = 1.0 + BETA * acc.abs().mean()
        buf1 = acc / g1
        A1_local = buf1.abs().mean()
        
        # Iteration 2 (local) - use local estimate
        M1_local = A1_local / (1.0 - BETA * A1_local)
        gr1 = 1.0 + BETA * M1_local
        acc_temp = buf1 * gr1
        
        g2 = 1.0 + BETA * acc_temp.abs().mean()
        buf2 = acc_temp / g2
        A2_local = buf2.abs().mean()
        
        # Now do one collective with both buffers fused
        fused = torch.cat([buf2, A1_local.unsqueeze(0), A2_local.unsqueeze(0)])
        fused_reduced = xm.all_reduce(xm.REDUCE_SUM, fused)
        
        # Extract results
        acc = fused_reduced[:-2] / W
        A1_global = fused_reduced[-2] / W
        A2_global = fused_reduced[-1] / W
        
        # Apply global corrections
        M2 = A2_global / (1.0 - BETA * A2_global)
        gr2 = 1.0 + BETA * M2
        acc = acc * gr2
    
    # Final iteration (7th) - just normalize and all-reduce
    g = 1.0 + BETA * acc.abs().mean()
    buf = acc / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s