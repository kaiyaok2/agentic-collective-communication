
def evolved_p4202(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: s = 3*AR(x1) + 0.5*AR(x2) + 7*AR(x3) + 1.5*AR(x4)
    
    # Concatenate scaled inputs directly
    flat = torch.cat([x1 * 3.0, x2 * 0.5, x3 * 7.0, x4 * 1.5])
    
    # Single all_reduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # View as (4, N) and sum
    s = reduced.view(4, N).sum(0)
    
    return s
