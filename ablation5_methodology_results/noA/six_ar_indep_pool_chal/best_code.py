
def evolved_p6603(x1, x2, x3, x4, x5, x6, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    stacked = torch.stack([x1, x2, x3, x4, x5, x6], dim=0)
    local_sum = stacked.sum(dim=0)
    result = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return result
