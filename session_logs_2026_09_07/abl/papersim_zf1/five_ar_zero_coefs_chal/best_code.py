
def evolved_p6502(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # y = 3*AR(x1) + 5*AR(x3) + 2*AR(x5)
    # Use linearity: AR(a + b + c) = AR(a) + AR(b) + AR(c)
    # Combine locally first, then single all_reduce on N elements instead of 3N
    local_sum = 3*x1 + 5*x3 + 2*x5
    return xm.all_reduce(xm.REDUCE_SUM, local_sum)
