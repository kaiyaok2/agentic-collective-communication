def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    r_idx = torch.arange(W, device=x.device, dtype=torch.long)
    a_tensor = 1.0 + 0.5 * (r_idx % 3).to(x.dtype)
    
    # Reshape and broadcast multiply
    result = s1.view(W, S) * a_tensor.unsqueeze(1)
    return result.flatten()