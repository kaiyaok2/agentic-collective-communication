
def evolved_p6100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute L2 norm squared globally: sum_i,r (x_r[i]^2)
    # 1. Square locally
    sq = x * x
    # 2. Sum locally to scalar
    local_sum = sq.sum()
    # 3. All-reduce the scalar (1 element instead of N)
    global_sum = xm.all_reduce(xm.REDUCE_SUM, local_sum)
    return global_sum
