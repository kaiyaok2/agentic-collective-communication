
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    for iteration in range(7):
        # Compute factors: max(1, 1 - mean)
        factors_1d = torch.clamp(1.0 - s.mean(dim=1), min=1.0)
        factors_2d = factors_1d.unsqueeze(1)
        
        # All-reduce with factors applied
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * factors_2d).view(-1)).view(B, S)
        
        # Normalize
        if iteration < 6:
            s = acc / (world_size * factors_2d)
        else:
            s = acc / world_size
    
    return s.view(-1)
