
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to (B, S) for block operations
    s_reshaped = s.view(B, S)
    
    # Compute factors for all blocks at once: 1 + abs(mean())
    factors = 1.0 + s_reshaped.mean(dim=1).abs()
    
    # Apply factors
    buf = s_reshaped * factors.unsqueeze(1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    return acc / world_size
