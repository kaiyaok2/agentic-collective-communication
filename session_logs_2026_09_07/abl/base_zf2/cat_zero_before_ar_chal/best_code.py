def evolved_p5500(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    padded = torch.cat([x, torch.zeros_like(x)])
    return xm.all_reduce(xm.REDUCE_SUM, padded)      # transfers 2N bytes
