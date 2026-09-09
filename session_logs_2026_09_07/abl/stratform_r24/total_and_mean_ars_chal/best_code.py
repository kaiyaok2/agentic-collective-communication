
def evolved_p6701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 3: Pre-scaled local fusion
    scale_factor = 1.0 + 1.0 / world_size
    scaled_x = x * scale_factor
    result = xm.all_reduce(xm.REDUCE_SUM, scaled_x)
    return result
