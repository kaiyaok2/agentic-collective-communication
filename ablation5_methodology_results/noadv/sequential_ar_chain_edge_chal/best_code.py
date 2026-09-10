
def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    x_size = x.numel()
    
    # Pack tensors
    packed = torch.cat([x.reshape(-1), y.reshape(-1)])
    
    # Single all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
    
    # Extract and compute in one step
    ax = reduced[:x_size]
    ay = reduced[x_size:]
    
    # Compute result and reshape
    result = ay + (ax * (2 * world_size))
    return result.view(y.shape)
