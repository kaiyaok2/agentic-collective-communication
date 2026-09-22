def r1_fold_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute all scaling factors once
    a_over_w = torch.tensor([1.0 + 0.5 * (r % 3) for r in range(W)], dtype=dtype) / W
    inv_w = 1.0 / W
    
    # Stage 1: First all_reduce
    ar1_out = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Stage 2: Vectorized scaling and second all_reduce
    ar1_out_reshaped = ar1_out.view(W, S)
    buf2_in = (ar1_out_reshaped * a_over_w.unsqueeze(1)).view(-1)
    ar2_out = xm.all_reduce(xm.REDUCE_SUM, buf2_in)
    
    # Stage 3: Vectorized final scale for third all_reduce
    buf3_in = ar2_out * inv_w
    ar3_out = xm.all_reduce(xm.REDUCE_SUM, buf3_in)
    
    return ar3_out