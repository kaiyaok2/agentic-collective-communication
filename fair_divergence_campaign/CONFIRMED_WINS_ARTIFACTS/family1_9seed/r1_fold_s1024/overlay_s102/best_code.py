def r1_fold_s1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute scaling factors as a tensor for vectorized operations
    a = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # First all_reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First local scaling: buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    # Vectorized: reshape and broadcast
    buf_reshaped = buf.view(W, S)
    scaling_1 = (a / W).view(W, 1)
    buf = (buf_reshaped * scaling_1).view(-1)
    
    # Second all_reduce
    buf = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Second and third local scaling fused
    # buf[r*S:(r+1)*S] = (a[r] / W) * buf[r*S:(r+1)*S] / max(a[r], 1e-9)
    # Vectorized
    buf_reshaped = buf.view(W, S)
    a_safe = torch.clamp(a, min=1e-9)
    scaling_2 = (a / (W * a_safe)).view(W, 1)
    buf = (buf_reshaped * scaling_2).view(-1)
    
    # Third all_reduce
    out = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return out