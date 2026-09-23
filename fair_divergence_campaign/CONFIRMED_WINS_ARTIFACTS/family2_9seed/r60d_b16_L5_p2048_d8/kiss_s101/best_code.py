def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 16; OFF = 2
    
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create mask for selection
    mask = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
