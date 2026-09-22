
def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size; D = 2
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    A_tot = sum(1.0 + 0.5*((k*11) % 7)/7.0 for k in range(W))
    u = 0.9 * A_tot
    # Try single iteration with adjusted scaling
    scale_squared = (a / u) * (A_tot / u)
    return xm.all_reduce(xm.REDUCE_SUM, scale_squared * x)
