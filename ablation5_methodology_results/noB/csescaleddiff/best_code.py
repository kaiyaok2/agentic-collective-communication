
def csescaleddiff_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Common subexpression elimination: all operations use the same all_reduce
    # (1.0 + 2.0 - 1.0 + 3.0 + 4.0) = 9.0
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 9.0 * reduced
