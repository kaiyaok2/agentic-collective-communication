def evolved_p116(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack x and abs(x), pack into one AR: [x | abs(x)], then AR with SUM & MAX would need two ops.
    # Best: 1 AR SUM for mean + 1 AR MAX for max_abs. Do them fused (single mark_step).
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    m = xm.all_reduce(xm.REDUCE_MAX, x.abs())
    return (s / world_size) / (m + 1e-8)
