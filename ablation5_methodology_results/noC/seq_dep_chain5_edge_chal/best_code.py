def evolved_p3900(x1, x2, x3, x4, x5, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Stack all tensors into a single tensor to reduce from 5 all_reduce calls to 1
    combined = torch.stack([x1, x2, x3, x4, x5], dim=0)
    
    # Single all_reduce operation
    reduced = xm.all_reduce(xm.REDUCE_SUM, combined)
    
    # Extract and combine with coefficients
    s = reduced[0] + 2 * reduced[1] + 3 * reduced[2] + 4 * reduced[3] + 5 * reduced[4]
    
    return s