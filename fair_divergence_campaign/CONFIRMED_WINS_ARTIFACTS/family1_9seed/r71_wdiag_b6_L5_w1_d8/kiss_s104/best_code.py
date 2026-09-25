
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute normalization factors A
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 6
        for j in range(5):
            A[(st + j) % 6] += w
    
    # Create normalization tensor (replicate each factor S times)
    A_flat = []
    for a in A:
        A_flat.extend([a] * S)
    A_tensor = torch.tensor(A_flat, device=x.device, dtype=x.dtype)
    
    # Precompute which block this rank excludes
    start = rank % 6
    exclude = (start - 1) % 6
    w = 0.5 + 0.02 * rank
    
    # 7 iterations
    for iteration in range(7):
        buf = w * s
        buf[exclude*S:(exclude+1)*S] = 0.0
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize (except on last iteration) - vectorized
        if iteration < 6:
            acc = acc / A_tensor
        s = acc
    
    return s
