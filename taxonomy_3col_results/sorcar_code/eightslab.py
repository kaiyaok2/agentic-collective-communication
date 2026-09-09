def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    
    # Single all_reduce on entire tensor
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Use broadcasting: reshape to (8, 1024), multiply by (8, 1), then flatten
    reshaped = reduced.view(n_slabs, slab_N)
    multipliers = torch.arange(1, n_slabs + 1, device=x.device, dtype=x.dtype).view(n_slabs, 1)
    result = reshaped * multipliers
    
    return result.flatten()