def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Mathematical optimization:
    # result = sum_r (y_r + 2 * sum_s x_s)
    #        = sum_r y_r + world_size * 2 * sum_s x_s
    
    # Stack x and y to do both all-reduces in one collective
    stacked = torch.stack([x, y], dim=0)  # (2, N)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)  # (2, N)
    
    sum_x = reduced[0]  # (N,)
    sum_y = reduced[1]  # (N,)
    
    # Compute result
    result = sum_y + world_size * 2 * sum_x
    
    return result