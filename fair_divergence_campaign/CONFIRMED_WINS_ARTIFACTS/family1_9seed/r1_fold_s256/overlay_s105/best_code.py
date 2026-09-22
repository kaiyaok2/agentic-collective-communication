def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute coefficient array 'a' for all ranks (globally determined)
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Create coefficient tensors for vectorized operations
    # Each coefficient is repeated S times for its corresponding shard
    a_tensor = torch.tensor([a[r] for r in range(W) for _ in range(S)], 
                           dtype=dtype, device=x.device)
    a_inv_tensor = torch.tensor([1.0 / max(a[r], 1e-9) for r in range(W) for _ in range(S)],
                                dtype=dtype, device=x.device)
    
    # Precompute sum of all coefficients and other constants
    sum_a = sum(a)
    a_scaled_tensor = a_tensor / W
    
    # Stage 1: First all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse stages 2-4: compute all local operations before next collective
    # Stage 2 computation: a[r] * s1 / W
    buf0 = a_scaled_tensor * s1
    
    # Stage 2 all_reduce
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Fuse stages 3-4 locally
    # Stage 3: s2 / a[r]
    s3 = s2 * a_inv_tensor
    # Stage 4: a[r] * s3 / W
    bufN = a_scaled_tensor * s3
    
    # Final all_reduce
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out