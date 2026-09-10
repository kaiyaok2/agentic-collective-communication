
def thirtytwoalt_fn(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # The original computes: 1*AR - 2*AR + 3*AR - 4*AR + ... + 31*AR - 32*AR
    # where AR = all_reduce(x)
    # This simplifies to: (1-2+3-4+...+31-32) * AR = -16 * AR
    result = -16.0 * xm.all_reduce(xm.REDUCE_SUM, x)
    return result
