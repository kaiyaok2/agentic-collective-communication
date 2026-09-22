
def r47_dd_meanabs_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with factor-based division
    for iteration in range(6):
        reshaped = s.view(B, S)
        f_tensor = 1.0 + reshaped.mean(dim=1).abs()
        f_expanded = f_tensor.unsqueeze(1).expand(B, S).reshape(-1)
        buf = s * f_expanded
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        divisor = world_size * f_expanded
        s = acc / divisor
    
    # 7th iteration: apply factors but divide only by world_size
    reshaped = s.view(B, S)
    f_tensor = 1.0 + reshaped.mean(dim=1).abs()
    f_expanded = f_tensor.unsqueeze(1).expand(B, S).reshape(-1)
    buf = s * f_expanded
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc / world_size
    
    return s
