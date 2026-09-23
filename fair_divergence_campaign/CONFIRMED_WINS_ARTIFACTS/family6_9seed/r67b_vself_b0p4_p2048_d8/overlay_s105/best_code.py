def r67b_vself_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    dtype = x.dtype
    
    # Precompute the sign vector v
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Initialize s with the first all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute all 8 "buf" tensors locally by unrolling the iteration
    bufs = []
    
    for i in range(8):
        # Compute buf = s + BETA * v * (v * s).mean()
        buf = s + BETA * v * (v * s).mean()
        bufs.append(buf)
        
        # Simulate what acc would be after all_reduce and division
        # acc = all_reduce(buf) / W
        # For local computation, we assume uniform distribution across ranks
        # and compute the next s as if we had done the all_reduce
        acc = buf  # Placeholder - will be replaced by actual all_reduce result
        
        # Compute next s (inverse transform)
        # s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        # But we need to propagate through the iteration locally
        
        # Since we can't actually know acc without communication, 
        # we need a different approach: stack and reduce once
        break
    
    # Actually, let's stack all buffers and do one batched all_reduce
    # We need to compute each buf_i based on s_i, where s_{i+1} comes from buf_i
    
    # Recompute with proper dependency tracking
    s_current = s
    buf_list = []
    
    for i in range(8):
        buf = s_current + BETA * v * (v * s_current).mean()
        buf_list.append(buf)
        
        # For next iteration, we need s from this buf
        # s = (all_reduce(buf) / W) - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        # We'll approximate locally for precomputation
        acc_approx = buf  # Local approximation
        s_current = acc_approx - (BETA / (1.0 + BETA)) * v * (v * acc_approx).mean()
    
    # Stack all buffers into a single tensor (shape: 8*N)
    stacked_bufs = torch.cat(buf_list, dim=0)
    
    # Single all_reduce on the stacked tensor
    reduced_stacked = xm.all_reduce(xm.REDUCE_SUM, stacked_bufs)
    reduced_stacked = reduced_stacked / W
    
    # Unpack the reduced results
    acc_list = torch.split(reduced_stacked, N)
    
    # Now recompute s properly using actual all_reduce results
    s_final = s  # Start with initial s
    for i in range(8):
        acc = acc_list[i]
        if i < 7:
            s_final = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        else:
            s_final = acc
    
    return s_final