
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 8; W = world_size; L = 2; OFF = 2
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create mask once for this rank's window
    start = (rank + OFF) % B
    mask = torch.zeros_like(s)
    for j in range(L):
        b = (start + j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Create normalization divisors tensor once
    norm = torch.ones_like(s)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = c[b]
    
    # Try 1 iteration (no normalization on last)
    buf = s * mask
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc
