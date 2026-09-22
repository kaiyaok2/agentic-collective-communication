
def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Sum across all ranks
    summed = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create full scale tensor directly
    scale_data = []
    for r in range(W):
        scale_data.extend([1.0 + 0.5*(r % 3)] * S)
    scale = torch.tensor(scale_data, device=x.device, dtype=x.dtype)
    result = summed * scale
    
    return result
