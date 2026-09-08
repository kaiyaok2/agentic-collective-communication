
def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise max across ranks of max(x, y, z)
    # Compute local max: max(x, y, z) = (x + y + z + |x-y| + |x-z| + |y-z|) / 2
    # Actually, simpler: use torch.stack and max along dim
    stacked = torch.stack([x, y, z], dim=0)
    local_max = stacked.max(dim=0)[0]
    # Then all-reduce with MAX operation
    result = xm.all_reduce(xm.REDUCE_MAX, local_max)
    return result
