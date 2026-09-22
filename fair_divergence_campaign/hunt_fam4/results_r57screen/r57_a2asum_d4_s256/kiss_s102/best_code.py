def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5*((k*11) % 7)/7.0 for k in range(world_size))
    u = 0.9 * A_tot
    
    scale = (A_tot ** 3) / (u ** 4)
    result = xm.all_reduce('sum', a * x) * scale
    return result