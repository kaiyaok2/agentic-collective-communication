
def sixtyfourinline_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Instead of 64 identical all_reduce operations summed together,
    # do one all_reduce and multiply by 64
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    return 64.0 * result
