def evolved_p3801(x, y, z, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack tensors (assuming same size)
    packed = torch.stack([x.flatten(), y.flatten(), z.flatten()])
    reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
    
    # Compute directly from stacked result
    return (2 * reduced[0] + 3 * reduced[1] + 5 * reduced[2]).view(x.shape)