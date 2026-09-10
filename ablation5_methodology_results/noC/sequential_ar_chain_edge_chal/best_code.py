def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine x and y into single tensor for one all_reduce
    combined = torch.stack([x, y])
    reduced = xm.all_reduce(xm.REDUCE_SUM, combined)
    sum_x = reduced[0]
    sum_y = reduced[1]
    scale = 2 * world_size
    return sum_y + sum_x * scale