
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Iterations 1-4
    for _ in range(4):
        f = (1.0 + torch.clamp(s.mean(dim=1), min=0.0)).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(B, S)
        s = s / (world_size * f)
    
    # Iteration 5
    f = (1.0 + torch.clamp(s.mean(dim=1), min=0.0)).unsqueeze(1)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)) / world_size
