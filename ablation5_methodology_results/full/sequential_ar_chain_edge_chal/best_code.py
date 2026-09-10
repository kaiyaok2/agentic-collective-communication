
def evolved_p3600(x, y, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    packed = torch.stack([x.reshape(-1), y.reshape(-1)])
    reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
    sx = reduced[0].reshape(x.shape)
    sy = reduced[1].reshape(y.shape)
    return sy + 2 * world_size * sx
