def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(B, S)  # Keep in reshaped form throughout
    
    for iter_idx in range(7):
        # Compute factors: 1.0 + max(0, -mean(dim=1))
        neg_means = -s.mean(dim=1)  # Shape: (B,)
        factors = 1.0 + (neg_means + neg_means.abs()) / 2
        
        # Apply factors and all-reduce
        buf = s * factors.unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        
        # Normalize
        if iter_idx < 6:
            s = s / (world_size * factors.unsqueeze(1))
        else:
            s = s / world_size
    
    return s.view(-1)