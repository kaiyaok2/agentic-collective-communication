
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A (normalization weights) - pure Python
    A = [0.0]*5
    for r in range(W):
        w = 0.5 + 0.02*r
        st = r % 5
        for j in range(3):
            A[(st + j) % 5] += w
    
    # Precompute weight mask once
    w = 0.5 + 0.02*rank
    start = rank % 5
    keep = set((start + j) % 5 for j in range(3))
    
    # Create weight mask tensor once
    weight_mask = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            weight_mask[b*S:(b+1)*S] = w
    
    # Create normalization tensor once
    norm_vec = torch.zeros_like(s)
    for b in range(5):
        norm_vec[b*S:(b+1)*S] = 1.0 / A[b]
    
    # 7 iterations
    for iteration in range(7):
        buf = s * weight_mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            s = acc * norm_vec
        else:
            s = acc
    
    return s
