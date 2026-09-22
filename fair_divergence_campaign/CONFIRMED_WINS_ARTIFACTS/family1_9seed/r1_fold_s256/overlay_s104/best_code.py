def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Pre-compute scaling factors as a tensor for broadcasting
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    a_tensor = torch.tensor(a, dtype=dtype, device=x.device)
    a_inv_tensor = torch.tensor([1.0 / max(val, 1e-9) for val in a], dtype=dtype, device=x.device)
    
    # First all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape s1 to (W, S) for broadcasting
    s1_reshaped = s1.view(W, S)
    
    # Apply per-shard scaling: buf0[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W
    # Broadcasting a_tensor (W,) with s1_reshaped (W, S) -> (W, S)
    buf0_reshaped = (a_tensor.unsqueeze(1) * s1_reshaped) / W
    buf0 = buf0_reshaped.view(W * S)
    
    # Second all-reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Reshape for inverse scaling: s1[r*S:(r+1)*S] = s1[r*S:(r+1)*S] / max(a[r], 1e-9)
    s1_reshaped = s1.view(W, S)
    s1_reshaped = a_inv_tensor.unsqueeze(1) * s1_reshaped
    s1 = s1_reshaped.view(W * S)
    
    # Apply per-shard scaling again: bufN[r*S:(r+1)*S] = a[r] * s1[r*S:(r+1)*S] / W
    s1_reshaped = s1.view(W, S)
    bufN_reshaped = (a_tensor.unsqueeze(1) * s1_reshaped) / W
    bufN = bufN_reshaped.view(W * S)
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out