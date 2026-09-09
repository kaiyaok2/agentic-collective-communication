
def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 3*AR(x1) + 0.5*AR(x2) + 7*AR(x3) + 1.5*AR(x4)
    # Pre-scale and stack for single all_reduce
    stacked = torch.stack([3 * x1, 0.5 * x2, 7 * x3, 1.5 * x4])
    reduced_stacked = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    s = reduced_stacked.sum(dim=0)
    return s
