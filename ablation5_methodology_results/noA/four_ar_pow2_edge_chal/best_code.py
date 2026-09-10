
def evolved_p4602(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Apply weights before stacking
    stacked = torch.stack([x1, 2 * x2, 4 * x3, 8 * x4], dim=0)
    
    # Single all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    
    # Sum all components
    result = reduced.sum(dim=0)
    return result
