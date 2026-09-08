
def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y_r = concat(x_{(r+1) % W}, x_{(r+2) % W})
    
    # Gather all vectors
    g = xm.all_gather(x, dim=0)
    
    # Compute offsets
    offset1 = ((rank + 1) % world_size) * N
    offset2 = ((rank + 2) % world_size) * N
    
    # If the two neighbors are contiguous in memory, we can extract in one shot
    if offset2 == offset1 + N:
        # Contiguous case: extract both at once
        return g.narrow(0, offset1, 2 * N)
    else:
        # Non-contiguous case (wraps around): need to cat
        return torch.cat([g.narrow(0, offset1, N), g.narrow(0, offset2, N)], dim=0)
