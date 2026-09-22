def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size; D = 2
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5*((k*11) % 7)/7.0 for k in range(W))
    u = 0.9 * A_tot
    cur = x
    for _t in range(D):
        y = xm.all_to_all(a * cur, split_dimension=0, concat_dimension=0,
                          split_count=W)
        z = torch.sum(y.reshape(W, S), dim=0)
        cur = xm.all_gather(z, dim=0) / u
    return cur
