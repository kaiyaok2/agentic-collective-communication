
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute accumulated weights for each bucket
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 6
        for j in range(5):
            A[(st + j) % 6] += w
    
    # Create normalization tensor using cat
    norm_parts = [torch.full((S,), 1.0/A[b], device=s.device, dtype=s.dtype) for b in range(6)]
    norm = torch.cat(norm_parts)
    
    # Pre-compute this rank's keep set and weight
    start = rank % 6
    keep_set = set((start + j) % 6 for j in range(5))
    zero_bucket = None
    for b in range(6):
        if b not in keep_set:
            zero_bucket = b
            break
    w = 0.5 + 0.02 * rank
    
    # Perform 7 rounds
    for round_idx in range(7):
        buf = w * s
        buf[zero_bucket*S:(zero_bucket+1)*S] = 0
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if round_idx < 6:
            acc = acc * norm
        
        s = acc
    
    return s
