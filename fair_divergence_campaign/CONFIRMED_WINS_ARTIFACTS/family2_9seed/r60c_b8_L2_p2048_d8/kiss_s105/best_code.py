
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 8
    W = world_size
    L = 2
    OFF = 2
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Pre-compute mask for this rank's window
    mask = torch.zeros_like(s)
    start = (rank + OFF) % B
    for j in range(L):
        b = (start + j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Pre-compute normalization factors (avoid division by zero)
    norm = torch.zeros_like(s)
    for b in range(B):
        if c[b] > 0:
            norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # 7 iterations: 6 with normalization, 1 without
    for i in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if i < 6:
            s = acc * norm
        else:
            s = acc
    
    return s
