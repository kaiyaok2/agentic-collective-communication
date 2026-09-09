
def evolved_p4101(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Pre-scale and concatenate
    scaled = torch.cat([x1, -0.5 * x2, 2.5 * x3, -1.5 * x4, 3.0 * x5])
    
    # Single all-reduce
    r = xm.all_reduce(xm.REDUCE_SUM, scaled).view(5, N)
    
    # Sum across segments
    return r.sum(dim=0)
