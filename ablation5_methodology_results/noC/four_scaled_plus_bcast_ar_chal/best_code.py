
def evolved_p6702(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Single all_reduce instead of 5
    sum_x = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute: (2+3+5+7)*sum_x = 17*sum_x
    result = 17 * sum_x
    
    # Add the ones reduction result (each element = world_size)
    ones_reduced = world_size * torch.ones_like(x)
    
    return result + ones_reduced
