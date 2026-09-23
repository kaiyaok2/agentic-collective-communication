
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    # Initial all-reduce and reshape to (B, S)
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Perform 7 rounds of computation
    for round_idx in range(7):
        # Compute factors for all batches: shape (B, 1)
        f = (1.0 + 2.0 * s.abs().mean(dim=1)).unsqueeze(1)
        
        # Scale, all-reduce (preserves shape), and descale
        acc = xm.all_reduce(xm.REDUCE_SUM, s * f)
        
        # Descale based on round
        if round_idx < 6:
            s = acc / (world_size * f)
        else:
            s = acc / world_size
    
    return s.view(-1)
