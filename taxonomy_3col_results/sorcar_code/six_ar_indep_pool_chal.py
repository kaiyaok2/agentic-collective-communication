
def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack and sum in one operation
    stacked = torch.stack([x1, x2, x3, x4, x5, x6], dim=0)
    local_sum = torch.sum(stacked, dim=0)
    # Then perform a single all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
