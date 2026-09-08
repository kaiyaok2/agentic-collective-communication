
def evolved_p7301(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: Per-row MIN all-reduce on x (32, 2048)
    
    return xm.all_reduce(xm.REDUCE_MIN, x)
