
def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute accumulated weights for normalization
    A = [0.0] * 6
    for r in range(W):
        w = 0.5 + 0.02 * r
        st = r % 6
        for j in range(5):
            A[(st + j) % 6] += w
    
    # Pre-compute which block to skip and weight for this rank
    skip_block = (rank - 1) % 6
    skip_start = skip_block * S
    skip_end = (skip_block + 1) * S
    w = 0.5 + 0.02 * rank
    
    # Create expanded normalization using concatenation
    norm_list = []
    for b in range(6):
        norm_list.append(torch.full((S,), 1.0 / A[b], device=x.device, dtype=x.dtype))
    norm_expanded = torch.cat(norm_list)
    
    # First 6 iterations with normalization
    for _ in range(6):
        buf = s * w
        buf[skip_start:skip_end] = 0
        s = xm.all_reduce(xm.REDUCE_SUM, buf) * norm_expanded
    
    # Final iteration without normalization
    buf = s * w
    buf[skip_start:skip_end] = 0
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
