def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try using stack + sum which might fuse better
    stacked = torch.stack([x1, x2 * 2, x3 * 4, x4 * 8], dim=0)
    local_sum = torch.sum(stacked, dim=0)
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result