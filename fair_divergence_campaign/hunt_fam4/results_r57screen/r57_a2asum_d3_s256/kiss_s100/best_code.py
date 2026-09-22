def r57_a2asum_d3_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    D = 3
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    inv_u = 1.0 / (0.9 * sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W)))
    scaled_a = a * inv_u
    
    cur = x
    for _t in range(D):
        cur = xm.all_reduce(xm.REDUCE_SUM, scaled_a * cur)
    
    return cur