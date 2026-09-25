
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*5
    for r in range(W):
        w_r = 0.5 + 0.02*r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w_r
    start = rank % 5
    keep = set((start + j) % 5 for j in range(3))
    w = 0.5 + 0.02*rank
    
    # Create mask and norm tensors using tensor operations
    mask = torch.zeros_like(s)
    norm = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            mask[b*S:(b+1)*S] = w
        norm[b*S:(b+1)*S] = 1.0/A[b]
    
    # 6 iterations with normalization
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * mask) * norm
    
    # Last iteration without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s
