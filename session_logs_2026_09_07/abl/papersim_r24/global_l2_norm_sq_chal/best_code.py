
def evolved_p6100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute global L2 norm squared: sum_i,r (x_r[i]^2)
    # Optimization: compute local sum of squares first, then all-reduce scalar
    # This reduces communication from N elements to 1 element
    local_sum = (x * x).sum()
    global_sum = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return global_sum
