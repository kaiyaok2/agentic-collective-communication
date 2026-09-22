
def r48_dd_relu_d6_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(8, 256)
    
    for _ in range(4):
        f = (1.0 + torch.clamp(s.mean(1), min=0.0)).unsqueeze(1)
        s = xm.all_reduce(xm.REDUCE_SUM, (s * f).view(-1)).view(8, 256) / (world_size * f)
    
    f = 1.0 + torch.clamp(s.mean(1), min=0.0)
    return xm.all_reduce(xm.REDUCE_SUM, (s * f.unsqueeze(1)).view(-1)) / world_size
