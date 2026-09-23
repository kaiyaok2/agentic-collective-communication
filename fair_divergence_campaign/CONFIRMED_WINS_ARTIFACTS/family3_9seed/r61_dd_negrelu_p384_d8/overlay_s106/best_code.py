def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 0-5: pipelined all-reduce with interleaved computation
    for iteration in range(6):
        # Compute factors for current iteration
        f = []
        for b in range(B):
            mb = -(s[b*S:(b+1)*S].mean())
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        # Prepare buffer for all-reduce
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        
        # Issue all-reduce immediately (communication starts)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Perform local computation (division by scaled factors)
        # This overlaps with communication from next iteration if pipelined
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        s = acc
    
    # Final iteration (iteration 6): compute factors, scale, and all-reduce
    f = []
    for b in range(B):
        mb = -(s[b*S:(b+1)*S].mean())
        f.append(1.0 + (mb if mb > 0 else mb*0.0))
    
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Final scaling by world_size only (no factor division)
    acc = acc / world_size
    
    s = acc
    return s