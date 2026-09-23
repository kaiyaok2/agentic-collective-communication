
def r61b_dd_absmean2_p768_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 768
    B = 8
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_view = s.view(B, S)
    
    # 6 iterations with scaling back
    for iteration in range(6):
        # Compute factors for all batches: f[b] = 1.0 + 2.0 * abs(s[b]).mean()
        abs_means = s_view.abs().mean(dim=1)  # Shape: (B,)
        f = 1.0 + 2.0 * abs_means
        
        # Scale by factors: buf[b] = s[b] * f[b]
        buf_view = s_view * f.unsqueeze(1)  # Broadcasting
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf_view.view(-1))
        acc_view = acc.view(B, S)
        
        # Scale down: acc[b] = acc[b] / (world_size * f[b])
        s_view = acc_view / (world_size * f.unsqueeze(1))
    
    # 7th iteration without scaling back by f
    abs_means = s_view.abs().mean(dim=1)
    f = 1.0 + 2.0 * abs_means
    buf_view = s_view * f.unsqueeze(1)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf_view.view(-1))
    acc = acc / world_size
    
    return acc
