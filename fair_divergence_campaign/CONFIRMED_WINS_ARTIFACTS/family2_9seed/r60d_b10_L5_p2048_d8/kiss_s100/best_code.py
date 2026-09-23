def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    
    # Compute which buckets this rank keeps (5 out of 10)
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(5))
    
    # Create a mask for kept buckets
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply mask and reduce
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s
