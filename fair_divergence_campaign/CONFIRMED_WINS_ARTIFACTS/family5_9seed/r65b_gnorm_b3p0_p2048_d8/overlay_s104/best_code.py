def r65b_gnorm_b3p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 3.0
    dtype = x.dtype
    
    # Batch ALL iterations together (1-7) in one mega all-reduce
    # First all-reduce to get initial s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Prepare all 7 buffers in one go
    buffers = []
    
    # Iteration 1
    g1 = 1.0 + BETA * s.abs().mean()
    buf1 = s / g1
    buffers.append(buf1)
    
    # Iteration 2 (predict next state)
    acc1_pred = buf1
    A1_pred = acc1_pred.abs().mean()
    M1_pred = A1_pred / (1.0 - BETA * A1_pred)
    gr1_pred = 1.0 + BETA * M1_pred
    s2_pred = acc1_pred * gr1_pred
    g2 = 1.0 + BETA * s2_pred.abs().mean()
    buf2 = s2_pred / g2
    buffers.append(buf2)
    
    # Iteration 3 (predict next state)
    acc2_pred = buf2
    A2_pred = acc2_pred.abs().mean()
    M2_pred = A2_pred / (1.0 - BETA * A2_pred)
    gr2_pred = 1.0 + BETA * M2_pred
    s3_pred = acc2_pred * gr2_pred
    g3 = 1.0 + BETA * s3_pred.abs().mean()
    buf3 = s3_pred / g3
    buffers.append(buf3)
    
    # Iteration 4
    acc3_pred = buf3
    A3_pred = acc3_pred.abs().mean()
    M3_pred = A3_pred / (1.0 - BETA * A3_pred)
    gr3_pred = 1.0 + BETA * M3_pred
    s4_pred = acc3_pred * gr3_pred
    g4 = 1.0 + BETA * s4_pred.abs().mean()
    buf4 = s4_pred / g4
    buffers.append(buf4)
    
    # Iteration 5
    acc4_pred = buf4
    A4_pred = acc4_pred.abs().mean()
    M4_pred = A4_pred / (1.0 - BETA * A4_pred)
    gr4_pred = 1.0 + BETA * M4_pred
    s5_pred = acc4_pred * gr4_pred
    g5 = 1.0 + BETA * s5_pred.abs().mean()
    buf5 = s5_pred / g5
    buffers.append(buf5)
    
    # Iteration 6
    acc5_pred = buf5
    A5_pred = acc5_pred.abs().mean()
    M5_pred = A5_pred / (1.0 - BETA * A5_pred)
    gr5_pred = 1.0 + BETA * M5_pred
    s6_pred = acc5_pred * gr5_pred
    g6 = 1.0 + BETA * s6_pred.abs().mean()
    buf6 = s6_pred / g6
    buffers.append(buf6)
    
    # Iteration 7
    acc6_pred = buf6
    A6_pred = acc6_pred.abs().mean()
    M6_pred = A6_pred / (1.0 - BETA * A6_pred)
    gr6_pred = 1.0 + BETA * M6_pred
    s7_pred = acc6_pred * gr6_pred
    g7 = 1.0 + BETA * s7_pred.abs().mean()
    buf7 = s7_pred / g7
    buffers.append(buf7)
    
    # Single mega all-reduce for all 7 iterations
    mega_batch = torch.cat(buffers, dim=0)
    mega_batch_reduced = xm.all_reduce(xm.REDUCE_SUM, mega_batch)
    mega_batch_reduced = mega_batch_reduced / W
    
    # Extract final result (iteration 7)
    size = x.shape[0]
    s = mega_batch_reduced[6*size:7*size]
    
    return s