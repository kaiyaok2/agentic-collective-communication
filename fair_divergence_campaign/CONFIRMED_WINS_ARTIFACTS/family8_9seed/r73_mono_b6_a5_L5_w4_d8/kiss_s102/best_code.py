
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Rank-specific values
    rank_w = 0.45 + 0.02 * (rank % 4)
    rank_start = (rank + 1) % 6
    rank_keep = [(rank_start + j) % 6 for j in range(5)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single iteration without normalization
    buf = torch.zeros_like(s)
    for b in rank_keep:
        jb = SIG[b]
        buf[b*S:(b+1)*S] = rank_w * s[jb*S:(jb+1)*S]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
