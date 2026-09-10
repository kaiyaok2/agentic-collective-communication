def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    
    # Create scale factors using repeat
    base = torch.arange(1, n_slabs + 1, device=x.device, dtype=x.dtype).view(n_slabs, 1)
    scale_tensor = base.repeat(1, slab_N).view(-1)
    
    # Scale then reduce
    result = xm.all_reduce(xm.REDUCE_SUM, x * scale_tensor)
    return result