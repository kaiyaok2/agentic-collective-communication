
def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Compute elementwise max across ranks of max(x, y, z)
    # Step 1: Compute local max of x, y, z using stack + max
    local_max = torch.stack([x, y, z], dim=0).max(dim=0)[0]
    # Step 2: All-reduce with REDUCE_MAX to get global max
    result = xm.all_reduce(xm.REDUCE_MAX, local_max)
    return result
