
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(7):
        # Compute all means at once using reshape
        s_reshaped = s.view(B, S)
        means = s_reshaped.mean(dim=1)
        f_tensor = 1.0 + means ** 2
        
        # Apply factors using broadcasting
        buf_reshaped = s_reshaped * f_tensor.unsqueeze(1)
        buf = buf_reshaped.view(-1)
        
        # All reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide
        if iteration < 6:
            f_list = f_tensor.tolist()
            scale = torch.cat([torch.full((S,), 1.0/(world_size * f_list[b]), 
                                         device=x.device, dtype=x.dtype) for b in range(B)])
            s = acc * scale
        else:
            s = acc / world_size
    
    return s
