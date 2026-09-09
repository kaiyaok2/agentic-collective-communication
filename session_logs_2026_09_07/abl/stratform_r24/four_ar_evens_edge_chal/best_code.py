
def evolved_p4700(x1, x2, x3, x4, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 5: Pairwise Batched Reduction
    pair1 = 2 * x1 + 4 * x2
    pair2 = 6 * x3 + 8 * x4
    batched = torch.cat([pair1, pair2], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, batched)
    s = reduced[:N] + reduced[N:2*N]
    return s
