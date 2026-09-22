
def r48_dd_square_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(7):
        means = s.mean(dim=1, keepdim=True)
        f_tensor = 1.0 + means**2
        
        acc = xm.all_reduce(xm.REDUCE_SUM, s * f_tensor)
        
        if iteration < 6:
            s = acc / (world_size * f_tensor)
        else:
            return (acc / world_size).view(-1)
    
    return s.view(-1)
