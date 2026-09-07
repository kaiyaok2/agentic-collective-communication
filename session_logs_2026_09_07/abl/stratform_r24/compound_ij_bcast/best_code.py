
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 4 Fixed: Reduce-scatter + All-gather
    x_flat = x.reshape(-1)
    
    # Reduce-scatter with SUM
    scattered = xm.reduce_scatter(xm.REDUCE_SUM, x_flat, scatter_dim=0, 
                                  shard_count=world_size, groups=None)
    
    # All-gather to reconstruct
    result_flat = xm.all_gather(scattered, dim=0)
    result = result_flat.reshape(N, N)
    return result
