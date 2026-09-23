
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(6):
        means = s.mean(dim=1, keepdim=True)
        factors_u = 1.0 + torch.where(-means > 0, -means, 0.0)
        s = xm.all_reduce(xm.REDUCE_SUM, s * factors_u) / (world_size * factors_u)
    
    means = s.mean(dim=1, keepdim=True)
    factors_u = 1.0 + torch.where(-means > 0, -means, 0.0)
    
    return xm.all_reduce(xm.REDUCE_SUM, (s * factors_u).view(-1)) / world_size
