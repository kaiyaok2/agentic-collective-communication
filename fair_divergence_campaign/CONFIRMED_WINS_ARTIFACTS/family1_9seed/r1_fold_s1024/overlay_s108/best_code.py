def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors as a tensor for vectorized operations
    a_list = [1.0 + 0.5 * (r % 3) for r in range(W)]
    a_tensor = torch.tensor(a_list, dtype=dtype, device=x.device).repeat_interleave(S)
    
    # First all-reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized first transformation: buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    buf = a_tensor * buf / W
    
    # Second all-reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Vectorized fused transformation: combine division by a[r] and multiplication by a[r]/W
    # buf[r*S:(r+1)*S] = (buf[r*S:(r+1)*S] / max(a[r], 1e-9)) * a[r] / W
    # This simplifies to: buf / W (since a[r] cancels out in numerator and denominator)
    a_tensor_safe = torch.clamp(a_tensor, min=1e-9)
    buf = (buf / a_tensor_safe) * a_tensor / W
    
    # Third all-reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return out