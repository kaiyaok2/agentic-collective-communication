
def evolved_p6000(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Combine 5 all_reduce operations into 1
    # 1.5 + 2.5 + 3.5 + 0.5 + 1.5 = 9.5
    return 9.5 * xm.all_reduce(xm.REDUCE_SUM, x)
