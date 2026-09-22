
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # This rank's window blocks
    start = (rank + OFF) % B
    keep_blocks = [(start + STR * j) % B for j in range(L)]
    
    # First iteration - inline masking without creating mask tensor
    buf = torch.zeros_like(s)
    for b in keep_blocks:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Normalize inline
    for b in range(B):
        if c[b] > 0:
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    
    # Second iteration - reuse buf
    buf = torch.zeros_like(s)
    for b in keep_blocks:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
