def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute constants as tensors for vectorized operations
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    # Create vectorized scaling tensors
    a_tensor = torch.zeros_like(x)
    a_inv_tensor = torch.zeros_like(x)
    for r in range(W):
        p = perm[r]
        a_tensor[p*S:(p+1)*S] = a_list[r] / W
        a_inv_tensor[p*S:(p+1)*S] = 1.0 / max(a_list[r], 1e-9)
    
    # First all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Iterations 1-6: Apply permutation, scale, reduce, inverse scale
    for iteration in range(6):
        # Apply permutation and scaling in one vectorized operation
        buf = torch.zeros_like(s)
        for r in range(W):
            p = perm[r]
            buf[p*S:(p+1)*S] = s[p*S:(p+1)*S]
        buf = buf * a_tensor
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Apply inverse permutation and scaling
        buf = torch.zeros_like(s)
        for r in range(W):
            p = perm[r]
            buf[p*S:(p+1)*S] = s[p*S:(p+1)*S]
        s = buf * a_inv_tensor
    
    # Final iteration (7th) - only forward pass
    buf = torch.zeros_like(s)
    for r in range(W):
        p = perm[r]
        buf[p*S:(p+1)*S] = s[p*S:(p+1)*S]
    buf = buf * a_tensor
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s