def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Pre-compute all scaling coefficients
    a_coeffs = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Fuse the scaling tensors to reduce memory allocation overhead
    # and prepare for potential optimization
    scale_tensor_1 = torch.zeros(W * S, dtype=dtype, device=x.device)
    scale_tensor_fused = torch.zeros(W * S, dtype=dtype, device=x.device)
    
    for r in range(W):
        # First scaling: a[r] / W
        scale_tensor_1[r*S:(r+1)*S] = a_coeffs[r] / W
        # Fused second and third scaling: (1 / max(a[r], 1e-9)) * (a[r] / W)
        scale_tensor_fused[r*S:(r+1)*S] = (1.0 / max(a_coeffs[r], 1e-9)) * (a_coeffs[r] / W)
    
    # Stage 1: all_reduce
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stage 2: vectorized element-wise multiply and all_reduce
    buf0 = s1 * scale_tensor_1
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf0)
    
    # Stage 3: fused multiply and all_reduce
    bufN = s2 * scale_tensor_fused
    out = xm.all_reduce(xm.REDUCE_SUM, bufN)
    
    return out