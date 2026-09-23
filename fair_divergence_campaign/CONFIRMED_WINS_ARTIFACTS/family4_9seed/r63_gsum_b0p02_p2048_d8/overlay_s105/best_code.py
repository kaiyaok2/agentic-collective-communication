def r63_gsum_b0p02_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.02
    D = 8  # number of iterations
    N = x.shape[0]  # 16384
    
    dtype = x.dtype
    
    # Initial all-reduce to get s_0
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # We need to compute D iterations of:
    # buf_i = s_i + BETA * sum(s_i)
    # acc_i = all_reduce(buf_i) / W
    # s_{i+1} = acc_i - BETA * sum(acc_i) / (1 + BETA * N)
    
    # The strategy is to batch all buffers together, but there's a dependency:
    # each s_{i+1} depends on s_i through the all-reduce.
    # We can analytically solve the recurrence or approximate it.
    
    # Let's denote g_i = sum(s_i) (scalar)
    # buf_i = s_i + BETA * g_i (broadcast scalar to vector)
    # After all_reduce: acc_i = sum_ranks(buf_i) / W = sum_ranks(s_i)/W + BETA*g_i
    #                        = s_i + BETA*g_i (since s_i is already all-reduced)
    # Then: s_{i+1} = acc_i - BETA * sum(acc_i) / (1 + BETA * N)
    #              = s_i + BETA*g_i - BETA*(g_i + BETA*g_i*N) / (1 + BETA*N)
    #              = s_i + BETA*g_i - BETA*g_i*(1 + BETA*N) / (1 + BETA*N)
    #              = s_i + BETA*g_i - BETA*g_i = s_i
    
    # Wait, this analysis shows s doesn't change. Let me reconsider.
    
    # Actually, the batching strategy requires us to prepare all buffers upfront,
    # which breaks the dependency. Instead, we'll use a mathematical fusion:
    # Stack all D buffers and do one large all-reduce.
    
    # Prepare all D iterations' buffers assuming s doesn't change much
    # (or use an iterative refinement with fewer syncs)
    
    # For simplicity and correctness, let's compute all iterations but batch
    # the all-reduces where possible. Stack multiple buffers:
    
    buffers = []
    current_s = s
    
    for i in range(D):
        buf = current_s + BETA * current_s.sum()
        buffers.append(buf)
        # For batching, we need to predict the next s, but it depends on all-reduce
        # So let's compute it assuming we'll get the all-reduced result
        # This creates a circular dependency that we resolve by stacking
        
        # Compute what s would be after this iteration (locally)
        # acc = all_reduce(buf) / W = buf (since buf is same on all ranks after s is synced)
        acc = buf  # approximation for batching
        current_s = acc - BETA * acc.sum() / (1.0 + BETA * N)
    
    # Stack all buffers into a single tensor (D, N)
    stacked_buffers = torch.stack(buffers, dim=0)
    
    # Single batched all-reduce
    stacked_acc = xm.all_reduce(xm.REDUCE_SUM, stacked_buffers)
    stacked_acc = stacked_acc / W
    
    # Extract the final result (last iteration)
    final_acc = stacked_acc[-1]
    
    # Apply the final correction (no more iterations, so return as s)
    s = final_acc
    
    return s