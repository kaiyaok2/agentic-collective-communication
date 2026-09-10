def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack tensors and sum along stack dimension
    stacked = torch.stack([x1, x2, x3, x4, x5, x6], dim=0)
    local_sum = torch.sum(stacked, dim=0)
    # Single all_reduce on the sum
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result