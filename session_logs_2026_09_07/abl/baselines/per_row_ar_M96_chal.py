def evolved_p6900(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    rows = [xm.all_reduce(xm.REDUCE_SUM, x[m]) for m in range(x.shape[0])]
    return torch.stack(rows, dim=0)
