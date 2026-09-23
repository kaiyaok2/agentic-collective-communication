
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    
    s_reshaped = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations with f[b] normalization
    for iteration in range(6):
        f = []
        for b in range(B):
            mb = -(s_reshaped[b].mean())
            f.append(1.0 + (mb if mb > 0 else mb*0.0))
        
        buf_reshaped = torch.zeros_like(s_reshaped)
        for b in range(B):
            buf_reshaped[b] = s_reshaped[b] * f[b]
        
        acc_reshaped = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1)).view(B, S)
        for b in range(B):
            acc_reshaped[b] = acc_reshaped[b] / (world_size * f[b])
        s_reshaped = acc_reshaped
    
    # 7th iteration without f[b] normalization
    f = []
    for b in range(B):
        mb = -(s_reshaped[b].mean())
        f.append(1.0 + (mb if mb > 0 else mb*0.0))
    
    buf_reshaped = torch.zeros_like(s_reshaped)
    for b in range(B):
        buf_reshaped[b] = s_reshaped[b] * f[b]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1)) / world_size
