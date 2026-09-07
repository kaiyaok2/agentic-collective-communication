
def evolved_p5602(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = AR(x1) + AR(x2) + AR(x3) + AR(x4) + AR(x5)
    # By linearity: = AR(x1+x2+x3+x4+x5)
    # Try stack+sum to see if it's more efficient
    stacked = torch.stack([x1, x2, x3, x4, x5])
    local_sum = stacked.sum(dim=0)
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
