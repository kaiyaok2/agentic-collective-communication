def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # 6 main iterations
    for iteration in range(6):
        f = 1.0 + torch.clamp(-s.mean(dim=1), min=0.0)
        f_broad = f.unsqueeze(1)
        acc = xm.all_reduce(xm.REDUCE_SUM, (s * f_broad).view(-1))
        s = acc.view(B, S) / (world_size * f_broad)
    
    # Final iteration
    f = 1.0 + torch.clamp(-s.mean(dim=1), min=0.0)
    acc = xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1))
    
    return acc / world_size