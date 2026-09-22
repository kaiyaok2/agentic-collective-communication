
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute counts for each bucket
    c = [0] * 5
    for r in range(world_size):
        st = r % 5
        c[st] += 1
        c[(st + 1) % 5] += 1
    
    # Pre-compute which buckets this rank keeps
    start = rank % 5
    keep = [start, (start + 1) % 5]
    
    # Create mask once
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create reciprocal tensor for division
    recip = torch.ones_like(s)
    for b in range(5):
        recip[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Do 6 rounds with division
    for _ in range(6):
        buf = s * mask
        s = xm.all_reduce(xm.REDUCE_SUM, buf) * recip
    
    # Final round without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
