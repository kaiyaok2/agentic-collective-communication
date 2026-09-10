
def mixedscaledseq_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine all the scaled all_reduce operations into one
    # 1.0 - 2.0 + 3.0 - 0.5 + 2.5 = 4.0
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    return 4.0 * reduced
