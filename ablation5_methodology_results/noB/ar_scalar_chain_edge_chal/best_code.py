def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack all inputs into one tensor to reduce collective operations
    stacked = torch.stack([x, y, z], dim=0)
    # Single all_reduce instead of 3 separate ones
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    # Extract and compute result
    return 5 * reduced[2] + 3 * reduced[1] + 2 * reduced[0]