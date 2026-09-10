def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack and sum in one operation
    stacked = torch.stack([x1, x2, x3, x4, x5], dim=0)
    combined = torch.sum(stacked, dim=0)
    result = xm.all_reduce(xm.REDUCE_SUM, combined)
    return result