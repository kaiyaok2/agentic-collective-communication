
def csescaleddiff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # CSE optimization: compute all_reduce once, then scale by the sum of coefficients
    # Original: 1.0 + 2.0 - 1.0 + 3.0 + 4.0 = 9.0
    t = xm.all_reduce(xm.REDUCE_SUM, x)
    acc = 9.0 * t
    return acc
