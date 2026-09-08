
def evolved_p6201(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack tensors and compute max along the stack dimension
    stacked = torch.stack([x, y, z], dim=0)
    local_max = torch.max(stacked, dim=0)[0]
    # All-reduce with REDUCE_MAX across all ranks
    result = xm.all_reduce(xm.REDUCE_MAX, local_max)
    return result
