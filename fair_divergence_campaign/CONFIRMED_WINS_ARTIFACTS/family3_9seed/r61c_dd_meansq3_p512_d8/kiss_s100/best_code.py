
def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with per-batch factor division
    for iteration in range(6):
        # Reshape for vectorized batch operations
        s_view = s.view(B, S)
        # Compute mean square for each batch: [B]
        mean_sq = (s_view * s_view).mean(dim=1)
        # Compute factors: [B]
        f = 1.0 + 3.0 * mean_sq
        # Broadcast to [B, S] and flatten to match s shape
        f_expanded = f.unsqueeze(1).expand(B, S).reshape(-1)
        # Multiply by factors
        buf = s * f_expanded
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Divide by world_size * factors
        s = acc / (world_size * f_expanded)
    
    # 7th iteration (different: no factor division, just world_size)
    s_view = s.view(B, S)
    mean_sq = (s_view * s_view).mean(dim=1)
    f = 1.0 + 3.0 * mean_sq
    f_expanded = f.unsqueeze(1).expand(B, S).reshape(-1)
    buf = s * f_expanded
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / world_size
    
    return s
