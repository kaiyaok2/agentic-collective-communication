
def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(6):
        # Reshape to (B, S) for vectorized operations
        s_blocks = s.view(B, S)
        
        # Compute factors for all blocks at once
        factors = 1.0 + s_blocks.mean(dim=1).abs()
        
        # Broadcast factors back to full shape
        scale_factors = factors.unsqueeze(1).expand(B, S).reshape(-1)
        
        # Apply factors
        scaled = s * scale_factors
        
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, scaled)
        
        # Unapply factors
        if iteration < 5:
            s = acc / (world_size * scale_factors)
        else:
            s = acc / world_size
    
    return s
