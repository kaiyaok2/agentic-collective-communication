def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all_reduce to get summed vector
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute all 6 iterations with pipelined all_reduces
    # Strategy: compute scaling factors for multiple iterations, 
    # then batch the all_reduce operations
    
    current = s
    
    # Batch iterations into groups to reduce collective dispatches
    # Group 1: iterations 0-2
    buffers_batch1 = []
    factors_batch1 = []
    
    for iteration in range(3):
        f = []
        for b in range(B):
            mb = current[b*S:(b+1)*S].mean()
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        buf = current.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = current[b*S:(b+1)*S] * f[b]
        
        buffers_batch1.append(buf)
        factors_batch1.append(f)
        
        if iteration < 2:
            simulated = buf.clone()
            for b in range(B):
                simulated[b*S:(b+1)*S] = buf[b*S:(b+1)*S] / f[b]
            current = simulated
    
    # Process batch 1 with combined buffer
    combined_buf1 = torch.stack(buffers_batch1)
    reduced_buf1 = xm.all_reduce(xm.REDUCE_SUM, combined_buf1)
    
    for iteration in range(3):
        acc = reduced_buf1[iteration]
        f = factors_batch1[iteration]
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        if iteration == 2:
            current = acc
    
    # Group 2: iterations 3-5
    buffers_batch2 = []
    factors_batch2 = []
    
    for iteration in range(3, 6):
        f = []
        for b in range(B):
            mb = current[b*S:(b+1)*S].mean()
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        buf = current.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = current[b*S:(b+1)*S] * f[b]
        
        buffers_batch2.append(buf)
        factors_batch2.append(f)
        
        if iteration < 5:
            simulated = buf.clone()
            for b in range(B):
                simulated[b*S:(b+1)*S] = buf[b*S:(b+1)*S] / f[b]
            current = simulated
    
    # Process batch 2 with combined buffer
    combined_buf2 = torch.stack(buffers_batch2)
    reduced_buf2 = xm.all_reduce(xm.REDUCE_SUM, combined_buf2)
    
    for iteration in range(3):
        acc = reduced_buf2[iteration]
        f = factors_batch2[iteration]
        if iteration < 2:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        else:
            acc = acc / world_size
    
    return acc