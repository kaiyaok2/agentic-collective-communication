
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Create coefficient tensor directly in expanded form
    a_list = []
    for r in range(W):
        coeff = 1.0 + 0.5*(r % 3)
        a_list.extend([coeff] * S)
    
    a_expanded = torch.tensor(a_list, device=x.device, dtype=x.dtype)
    a_scaled = a_expanded / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 iterations with inverse operation
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
        s = s / a_expanded
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
    
    return s
