def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack all tensors and sum along the new dimension
    combined = torch.stack([x1, x2, x3, x4, x5, x6], dim=0).sum(dim=0)
    # Single all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, combined)