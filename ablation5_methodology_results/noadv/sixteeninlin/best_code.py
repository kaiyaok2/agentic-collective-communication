
def sixteeninlin_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The original code does 16 identical all_reduce operations and sums them
    # This is equivalent to doing one all_reduce and multiplying by 16
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 16.0 * result
