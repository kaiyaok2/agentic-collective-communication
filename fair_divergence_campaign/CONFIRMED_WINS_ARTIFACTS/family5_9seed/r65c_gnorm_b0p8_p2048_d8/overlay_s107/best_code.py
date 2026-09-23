def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    device = x.device
    
    # Pre-allocate buffer for packing to avoid repeated allocations
    packed_buf = torch.empty(x.numel() + 1, dtype=dtype, device=device)
    
    # First all-reduce: initial sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6: Pack vector with scalar
    for _ in range(6):
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        A = buf.abs().mean()
        
        # Pack without creating new tensors
        packed_buf[:-1] = buf
        packed_buf[-1] = A
        
        packed_reduced = xm.all_reduce(xm.REDUCE_SUM, packed_buf)
        
        acc = packed_reduced[:-1] / W
        A_sum = packed_reduced[-1] / W
        M = A_sum / (1.0 - BETA * A_sum)
        gr = 1.0 + BETA * M
        s = acc * gr
    
    # Final iteration 7: Just reduce the final buffer
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / W
    
    return s