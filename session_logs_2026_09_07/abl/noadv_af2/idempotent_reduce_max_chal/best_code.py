
def evolved_p5103(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute element-wise max of x across all ranks
    # One all-reduce MAX is sufficient; subsequent MAX operations
    # are redundant due to idempotence: max(a, max(b, c)) = max(a, b, c)
    return xm.all_reduce(xm.REDUCE_MAX, x)
