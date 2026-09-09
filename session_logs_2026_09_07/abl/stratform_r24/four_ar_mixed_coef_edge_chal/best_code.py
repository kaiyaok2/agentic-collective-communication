
def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 5: Stacked 2D AR with pre-scaling
    stacked = torch.stack([3 * x1, 0.5 * x2, 7 * x3, 1.5 * x4], dim=0)
    ar_stacked = xm.all_reduce(xm.REDUCE_SUM, stacked)
    s = torch.sum(ar_stacked, dim=0)
    return s
