def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    parts = []
    for i in range(n_slabs):
        s = xm.all_reduce(xm.REDUCE_SUM, x[i*slab_N:(i+1)*slab_N])
        parts.append(s * (i + 1))
    return torch.cat(parts)
