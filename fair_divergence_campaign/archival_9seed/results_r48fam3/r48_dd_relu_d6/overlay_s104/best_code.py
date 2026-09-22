def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    dtype = x.dtype
    
    # First all-reduce: sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape once for all iterations
    s = s.view(B, S)
    
    # Repeat the pattern 5 times with fused operations
    for iteration in range(5):
        # Compute block means, ReLU, and factors in one pass
        mb = s.mean(dim=1, keepdim=True)  # Shape: (B, 1)
        f = 1.0 + torch.clamp(mb, min=0.0)  # Fused ReLU and addition
        
        # Scale each block by its factor using broadcasting
        buf = s * f  # Broadcasting: (B, S) * (B, 1)
        
        # All-reduce the scaled buffer (flatten inline)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
        
        # Divide by (world_size * factor) and reshape in one step
        s = (acc.view(B, S) / (world_size * f))
    
    # Final iteration (6th): compute factors, scale, all-reduce, then divide by world_size only
    mb = s.mean(dim=1, keepdim=True)
    f = 1.0 + torch.clamp(mb, min=0.0)
    buf = s * f
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    # Final division by world_size only (not by factors)
    return acc / world_size