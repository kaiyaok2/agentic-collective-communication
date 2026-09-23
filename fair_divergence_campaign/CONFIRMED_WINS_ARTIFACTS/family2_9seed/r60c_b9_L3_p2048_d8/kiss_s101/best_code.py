
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # Compute counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(3):
            b = (st + j) % B
            c[b] += 1
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep = [(start + j) % B for j in range(3)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single selective aggregation
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
