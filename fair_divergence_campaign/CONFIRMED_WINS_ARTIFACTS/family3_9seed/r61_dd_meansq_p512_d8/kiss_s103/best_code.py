
def r61_dd_meansq_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Work in reshaped (B, S) format throughout to minimize reshaping
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(7):
        # Compute factors for all batches
        f = 1.0 + (s * s).mean(dim=1)  # shape: (B,)
        
        # Apply factors
        buf = s * f.unsqueeze(1)
        
        # All reduce (flatten for collective, then reshape back)
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.reshape(-1)).view(B, S)
        
        # Normalize
        if iteration < 6:
            s = acc / (world_size * f.unsqueeze(1))
        else:
            s = acc / world_size
    
    return s.reshape(-1)
