
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute which blocks this rank keeps and weight
    start = rank % 5
    w = 0.5 + 0.02 * rank
    
    # Compute normalization factors A
    A = [0.0] * 5
    for r in range(W):
        w_r = 0.5 + 0.02 * r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w_r
    
    # Create combined mask-norm tensor for first 6 iterations
    mask_norm = torch.zeros(5 * S, device=x.device, dtype=x.dtype)
    for j in range(3):
        b = (start + j) % 5
        mask_norm[b*S:(b+1)*S] = w / A[b]
    
    # Create mask for final iteration
    mask_final = torch.zeros(5 * S, device=x.device, dtype=x.dtype)
    for j in range(3):
        b = (start + j) % 5
        mask_final[b*S:(b+1)*S] = w
    
    # First 6 iterations with combined mask and normalization
    for _ in range(6):
        buf = s * mask_norm
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # 7th iteration without normalization
    buf = s * mask_final
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
