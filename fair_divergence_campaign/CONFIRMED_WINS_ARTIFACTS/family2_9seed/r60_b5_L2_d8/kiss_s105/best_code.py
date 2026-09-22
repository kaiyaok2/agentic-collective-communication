
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute counts
    c = [0] * 5
    for r in range(W):
        st = r % 5
        c[(st) % 5] += 1
        c[(st + 1) % 5] += 1
    
    # Pre-compute which buckets this rank keeps
    start = rank % 5
    keep_list = [(start + j) % 5 for j in range(2)]
    drop_list = [b for b in range(5) if b not in keep_list]
    
    # 6 iterations with division
    for _ in range(6):
        buf = s.clone()
        for b in drop_list:
            buf[b*S:(b+1)*S] = 0.0
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        for b in range(5):
            s[b*S:(b+1)*S] = s[b*S:(b+1)*S] / c[b]
    
    # Final iteration without division
    buf = s.clone()
    for b in drop_list:
        buf[b*S:(b+1)*S] = 0.0
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
