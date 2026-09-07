
def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 3*AR(x1) + 0.5*AR(x2) + 7*AR(x3) + 1.5*AR(x4)
    # Since AR is linear: AR(c*x) = c*AR(x), we can pre-scale
    # s = AR(3*x1) + AR(0.5*x2) + AR(7*x3) + AR(1.5*x4)
    
    # Pre-scale all inputs
    stacked = torch.stack([3 * x1, 0.5 * x2, 7 * x3, 1.5 * x4])
    
    # Single all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    # Sum the reduced results
    return reduced.sum(dim=0)
