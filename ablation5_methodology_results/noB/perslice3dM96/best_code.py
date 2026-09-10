def perslice3dM96_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    batches = [xm.all_reduce(xm.REDUCE_SUM, x[b]) for b in range(x.shape[0])]
    return torch.stack(batches, dim=0)
