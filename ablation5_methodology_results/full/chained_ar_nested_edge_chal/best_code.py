
def evolved_p3701(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Only need one all_reduce
    ax1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Expand the math:
    # y1 = ax1 * 2
    # ax2 = ax1 + y1 * world_size = ax1 * (1 + 2 * world_size)
    # y2 = ax2 + 3
    # ax3 = ax1 + y2 * world_size = ax1 * (1 + (1 + 2*ws)*ws) + 3*ws
    
    ws = world_size
    coefficient = 1 + ws + 2 * ws * ws
    ax3 = ax1 * coefficient + 3 * ws
    
    return ax3
