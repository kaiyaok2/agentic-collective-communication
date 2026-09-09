
def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute MAX and MIN reductions once (x doesn't change)
    x_max = xm.all_reduce(xm.REDUCE_MAX, x)
    x_min = xm.all_reduce(xm.REDUCE_MIN, x)
    
    # Sum of coefficients:
    # MAX: 0.1 * (1+2+3+...+8) = 0.1 * 36 = 3.6
    # MIN: 0.05 * (1+2+3+...+8) = 0.05 * 36 = 1.8
    a = x_max * 3.6 + x_min * 1.8
    
    return a
