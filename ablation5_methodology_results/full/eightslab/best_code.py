
def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    
    # Try using unsqueeze + repeat + view
    weights = torch.arange(1, n_slabs + 1, device=x.device, dtype=x.dtype).unsqueeze(1).repeat(1, slab_N).view(-1)
    
    # Multiply and reduce
    weighted = x * weights
    result = xm.all_reduce(xm.REDUCE_SUM, weighted)
    
    return result
