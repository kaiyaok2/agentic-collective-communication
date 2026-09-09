
def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: a = sum over i=0..7 of [MAX_reduce(x) * (i+1)*0.1 + MIN_reduce(x) * (i+1)*0.05]
    # Simplifies to: MAX_reduce(x) * 3.6 + MIN_reduce(x) * 1.8
    # where 3.6 = 0.1 * sum(1..8) and 1.8 = 0.05 * sum(1..8)
    
    max_reduced = xm.all_reduce(xm.REDUCE_MAX, x)
    min_reduced = xm.all_reduce(xm.REDUCE_MIN, x)
    a = max_reduced * 3.6 + min_reduced * 1.8
    return a
