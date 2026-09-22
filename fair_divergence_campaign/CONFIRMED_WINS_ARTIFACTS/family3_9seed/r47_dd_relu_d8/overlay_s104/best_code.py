def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Strategy: Batch all 8 iterations into a single all-reduce
    # Prepare all buffers for iterations 1-7
    buffers = []
    current = s
    factors_list = []
    
    for iter_idx in range(7):
        # Compute scaling factors for current state
        f = []
        for b in range(B):
            mb = current[b*S:(b+1)*S].mean()
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        factors_list.append(f)
        
        # Create scaled buffer
        buf = current.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = current[b*S:(b+1)*S] * f[b]
        
        buffers.append(buf)
        
        # Speculatively compute next state
        acc = buf.clone()
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / f[b]
        current = acc
    
    # Compute final iteration buffer
    f_final = []
    for b in range(B):
        mb = current[b*S:(b+1)*S].mean()
        f_final.append(1.0 + (mb if mb > 0 else mb*0.0))
    buf_final = current.clone()
    for b in range(B):
        buf_final[b*S:(b+1)*S] = current[b*S:(b+1)*S] * f_final[b]
    buffers.append(buf_final)
    
    # Concatenate all 8 buffers and do single all-reduce
    mega_buffer = torch.cat(buffers, dim=0)
    mega_result = xm.all_reduce(xm.REDUCE_SUM, mega_buffer)
    
    # Split results back
    tensor_size = x.shape[0]
    results = [mega_result[i*tensor_size:(i+1)*tensor_size] for i in range(8)]
    
    # Process results iteratively
    s = s
    for idx in range(7):
        acc = results[idx]
        f = factors_list[idx]
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Final result (8th iteration)
    acc = results[7]
    acc = acc / world_size
    
    return acc