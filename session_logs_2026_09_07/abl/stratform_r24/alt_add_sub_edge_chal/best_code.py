
def evolved_p3902(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single stacked batch (4, N)
    stacked = torch.stack([x1, x2, x3, x4])  # Shape: (4, N)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    s = reduced[0] - reduced[1] + reduced[2] - reduced[3]
    return s
