
def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack x and y, reduce together
    # result = sum_r(y_r) + 2*world_size*sum_r(x_r)
    
    stacked = torch.stack([x, y])  # shape: (2, N)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)  # shape: (2, N)
    
    ax = reduced[0]  # sum_r x_r
    ay = reduced[1]  # sum_r y_r
    
    result = ay + (2 * world_size) * ax
    
    return result
