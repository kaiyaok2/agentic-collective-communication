def r48_dd_meansq_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-reduce to get the sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Batch ALL 8 iterations together
    bufs = []
    f_values = []
    
    # Prepare buffers for all 8 iterations
    for iteration in range(8):
        f = [1.0 + (s[b*S:(b+1)*S]**2).mean() for b in range(B)]
        buf = s.clone()
        for b in range(B):
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
        bufs.append(buf)
        f_values.append(f)
    
    # Concatenate all 8 buffers and do a single large all-reduce
    concat_buf = torch.cat(bufs, dim=0)
    concat_acc = xm.all_reduce(xm.REDUCE_SUM, concat_buf)
    
    # Unpack and process iterations 0-6 (with normalization)
    for i in range(7):
        acc = concat_acc[i*B*S:(i+1)*B*S]
        f = f_values[i]
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        s = acc
    
    # Process final iteration 7 - just divide by world_size
    acc = concat_acc[7*B*S:8*B*S]
    acc = acc / world_size
    s = acc
    
    return s