
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    SIG = [3, 2, 1, 0, 5, 4]
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply weight to entire tensor
    w = 0.45 + 0.02 * (rank % 4)
    s_weighted = w * s
    
    # Compute blocks to permute
    start = (rank + 1) % 6
    keep_blocks = [(start + j) % 6 for j in range(5)]
    
    # Apply permutation
    buf = torch.zeros_like(s)
    for b in keep_blocks:
        jb = SIG[b]
        buf[b * S:(b + 1) * S] = s_weighted[jb * S:(jb + 1) * S]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
