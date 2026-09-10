def evolved_p5502(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    if world_size < 2:
        return x + y + z
    
    # Optimized path: sum then reduce
    if x.shape == y.shape == z.shape:
        return xm.all_reduce(xm.REDUCE_SUM, x + y + z)
    
    # General case: concatenate, reduce, split
    reduced = xm.all_reduce(xm.REDUCE_SUM, 
                           torch.cat([x.view(-1), y.view(-1), z.view(-1)], dim=0))
    sx = x.numel()
    sy = y.numel()
    return (reduced[:sx].view(x.shape) + 
            reduced[sx:sx+sy].view(y.shape) + 
            reduced[sx+sy:].view(z.shape))