
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts
    c = [0] * 10
    for r in range(W):
        st = (r + 2) % 10
        ks = set((st + 1*j) % 10 for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Compute which buckets to keep
    B = 10
    OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Create mask
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create combined mask*norm for iterations 0-5
    mask_norm = torch.zeros_like(s)
    for b in range(10):
        if b in keep and c[b] > 0:
            mask_norm[b*S:(b+1)*S] = 1.0 / c[b]
    
    # First 6 iterations use mask_norm
    for iteration in range(6):
        buf = s * mask_norm
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Last iteration uses only mask (no normalization)
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
