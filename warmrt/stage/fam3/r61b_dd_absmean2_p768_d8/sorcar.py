
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations: full scale/unscale
    for iteration in range(6):
        # Compute means for all batches at once
        s_view = s.view(B, S)
        means = s_view.abs().mean(dim=1)  # [B]
        
        # Compute factors and expand to full size
        f_vec = 1.0 + 2.0 * means
        factors = f_vec.unsqueeze(1).expand(B, S).reshape(-1)
        
        acc = xm.all_reduce(xm.REDUCE_SUM, s * factors)
        s = acc / (world_size * factors)
    
    # 7th iteration: scale but only divide by world_size
    s_view = s.view(B, S)
    means = s_view.abs().mean(dim=1)
    f_vec = 1.0 + 2.0 * means
    factors = f_vec.unsqueeze(1).expand(B, S).reshape(-1)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, s * factors)
    s = acc / world_size
    
    return s
