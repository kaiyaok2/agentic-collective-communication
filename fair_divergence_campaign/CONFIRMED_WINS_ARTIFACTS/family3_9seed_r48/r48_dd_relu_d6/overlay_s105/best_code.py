def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # Initial all-gather to collect all ranks' data
    gathered = xm.all_gather(x, dim=0)  # Shape: (world_size * B * S,)
    
    # Reshape for easier per-rank processing
    # gathered[r*B*S:(r+1)*B*S] contains rank r's data
    
    # Initialize s with the sum across all ranks
    s = torch.zeros(B * S, dtype=dtype, device=x.device)
    for r in range(world_size):
        s += gathered[r * B * S:(r + 1) * B * S]
    
    # Perform 5 iterations of the transformation
    for iteration in range(5):
        # Compute block means and scaling factors
        f = []
        for b in range(B):
            mb = s[b * S:(b + 1) * S].mean()
            f.append(1.0 + (mb if mb > 0 else mb * 0.0))
        
        # Scale each block
        buf = s.clone()
        for b in range(B):
            buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S] * f[b]
        
        # All-gather the scaled buffer
        gathered_buf = xm.all_gather(buf, dim=0)  # Shape: (world_size * B * S,)
        
        # Sum across all ranks
        acc = torch.zeros(B * S, dtype=dtype, device=x.device)
        for r in range(world_size):
            acc += gathered_buf[r * B * S:(r + 1) * B * S]
        
        # Unscale by dividing by (world_size * f[b]) for each block
        for b in range(B):
            acc[b * S:(b + 1) * S] = acc[b * S:(b + 1) * S] / (world_size * f[b])
        
        s = acc
    
    # Final iteration (6th): compute scaling factors, scale, gather, sum, and divide by world_size
    f = []
    for b in range(B):
        mb = s[b * S:(b + 1) * S].mean()
        f.append(1.0 + (mb if mb > 0 else mb * 0.0))
    
    buf = s.clone()
    for b in range(B):
        buf[b * S:(b + 1) * S] = s[b * S:(b + 1) * S] * f[b]
    
    # All-gather the scaled buffer
    gathered_buf = xm.all_gather(buf, dim=0)
    
    # Sum across all ranks and divide by world_size
    acc = torch.zeros(B * S, dtype=dtype, device=x.device)
    for r in range(world_size):
        acc += gathered_buf[r * B * S:(r + 1) * B * S]
    
    acc = acc / world_size
    s = acc
    
    return s