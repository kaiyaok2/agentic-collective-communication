def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and A_inv
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 6
        for j in range(5):
            A[(st + j) % 6] += w
    A_inv = [1.0 / a for a in A]
    
    # Precompute which buckets this rank keeps and weight
    start = rank % 6
    keep_list = [(start + j) % 6 for j in range(5)]
    w = 0.5 + 0.02 * rank
    
    # Main loop
    for iteration in range(7):
        # Create buffer and fill using view operations
        buf = torch.zeros_like(s)
        s_view = s.view(6, S)
        buf_view = buf.view(6, S)
        
        for b in keep_list:
            buf_view[b] = w * s_view[b]
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize (except last iteration)
        if iteration < 6:
            acc_view = acc.view(6, S)
            for b in range(6):
                acc_view[b] *= A_inv[b]
        
        s = acc
    
    return s
