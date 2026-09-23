def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 16
    OFF = 2
    
    # Compute keep set for this rank
    start = (rank + OFF) % B
    keep_indices = [(start + j) % B for j in range(5)]
    
    # Compute counts for averaging
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(5):
            b = (st + j) % B
            c[b] += 1
    
    # Create mask tensor
    mask_flat = []
    for b in range(B):
        if b in keep_indices:
            mask_flat.extend([1.0] * S)
        else:
            mask_flat.extend([0.0] * S)
    mask = torch.tensor(mask_flat, device=x.device, dtype=x.dtype)
    
    # Create inverse count tensor
    inv_c_flat = []
    for b in range(B):
        val = 1.0 / c[b] if c[b] > 0 else 1.0
        inv_c_flat.extend([val] * S)
    inv_c = torch.tensor(inv_c_flat, device=x.device, dtype=x.dtype)
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * mask) * inv_c
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s * mask)
    
    return s