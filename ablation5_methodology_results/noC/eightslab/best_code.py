def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    # Create weight vector [1, 2, 3, 4, 5, 6, 7, 8]
    weight_vec = torch.arange(1, n_slabs+1, dtype=x.dtype, device=x.device)
    # Reshape x to (n_slabs, slab_N) for broadcasting
    x_reshaped = x.view(n_slabs, slab_N)
    # Multiply each slab by its weight: slab_i * (i+1)
    weighted = x_reshaped * weight_vec.view(n_slabs, 1)
    # Flatten back to original shape
    weighted_flat = weighted.view(-1)
    # Single all_reduce instead of 8
    return xm.all_reduce(xm.REDUCE_SUM, weighted_flat)