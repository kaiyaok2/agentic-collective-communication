
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape once for vectorized operations
    s = s.view(B, S)
    
    # 7 iterations of the refinement process
    for iteration in range(7):
        # Compute factors for all batches at once
        batch_means = s.abs().mean(dim=1)  # Shape: (B,)
        factors = 1.0 + 2.0 * batch_means
        
        # Apply factors: broadcast multiply
        weighted = s * factors.unsqueeze(1)
        
        # All-reduce (flatten for collective)
        acc = xm.all_reduce(xm.REDUCE_SUM, weighted.view(-1))
        acc = acc.view(B, S)
        
        # Normalize
        if iteration < 6:
            s = acc / (world_size * factors.unsqueeze(1))
        else:
            s = acc / world_size
    
    return s.view(-1)
