
def mixedscaledseq_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The original computes: 1.0 - 2.0 + 3.0 - 0.5 + 2.5 = 4.0
    # So we just need to do one all_reduce and multiply by 4.0
    result = 4.0 * xm.all_reduce(xm.REDUCE_SUM, x)
    return result
