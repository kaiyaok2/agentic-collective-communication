
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        # Compute all means at once
        reshaped = s.view(B, S)
        means = reshaped.mean(dim=1)  # Shape: (B,)
        
        # Apply transformation
        f = []
        buf = s.clone()
        for b in range(B):
            mb = -means[b]
            factor = 1.0 + (mb if mb > 0 else mb*0.0)
            f.append(factor)
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * factor
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        for b in range(B):
            acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
        
        s = acc
    
    # Final iteration
    reshaped = s.view(B, S)
    means = reshaped.mean(dim=1)
    
    buf = s.clone()
    for b in range(B):
        mb = -means[b]
        factor = 1.0 + (mb if mb > 0 else mb*0.0)
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * factor
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc / world_size
