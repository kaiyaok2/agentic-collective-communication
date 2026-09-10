
def mixedscaledseq_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Original: 1.0 * all_reduce(x) - 2.0 * all_reduce(x) + 3.0 * all_reduce(x) - 0.5 * all_reduce(x) + 2.5 * all_reduce(x)
    # Simplifies to: (1.0 - 2.0 + 3.0 - 0.5 + 2.5) * all_reduce(x) = 4.0 * all_reduce(x)
    return 4.0 * xm.all_reduce(xm.REDUCE_SUM, x)
