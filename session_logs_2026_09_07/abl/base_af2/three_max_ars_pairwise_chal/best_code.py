
def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute element-wise max across ranks of max(x, y, z)
    # First compute local max, then single all-reduce
    xy_max = torch.where(x > y, x, y)
    local_max = torch.where(xy_max > z, xy_max, z)
    result = xm.all_reduce(xm.REDUCE_MAX, local_max)
    return result
