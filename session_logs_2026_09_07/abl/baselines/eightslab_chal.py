def evolved_p9002(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    N = x.shape[0] // 8
    parts = [xm.all_reduce(xm.REDUCE_SUM, x[i*N:(i+1)*N]) for i in range(8)]
    return torch.cat(parts, dim=0)
