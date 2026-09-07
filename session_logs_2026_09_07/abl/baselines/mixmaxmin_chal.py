def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = torch.zeros_like(x)
    for i in range(8):
        a = a + xm.all_reduce(xm.REDUCE_MAX, x) * ((i + 1) * 0.1)
        a = a + xm.all_reduce(xm.REDUCE_MIN, x) * ((i + 1) * 0.05)
    return a
