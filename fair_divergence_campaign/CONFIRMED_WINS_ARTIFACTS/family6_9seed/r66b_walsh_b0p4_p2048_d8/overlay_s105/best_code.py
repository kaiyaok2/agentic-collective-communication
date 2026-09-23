def r66b_walsh_b0p4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.4
    N = 16384
    
    # Precompute fixed sign vectors and other constants
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    u = (1 - 2 * ((idx // 2) % 2)).to(x.dtype)
    
    # Precompute BETA * u to avoid repeated multiplications
    beta_u = BETA * u
    inv_W = 1.0 / W
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Helper function to compute the transformation with more local ops
    def transform_and_reduce(s_in):
        # Do more local computation before collective
        vs = v * s_in
        vs_mean = vs.mean()
        buf = s_in + beta_u * vs_mean
        
        # Single collective operation
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Post-processing with local ops
        acc = acc * inv_W
        v_acc = v * acc
        v_acc_mean = v_acc.mean()
        result = acc - beta_u * v_acc_mean
        
        return result
    
    # Apply 7 transformation stages
    # Unroll to ensure compiler can optimize
    s = transform_and_reduce(s)
    s = transform_and_reduce(s)
    s = transform_and_reduce(s)
    s = transform_and_reduce(s)
    s = transform_and_reduce(s)
    s = transform_and_reduce(s)
    
    # Final stage (no post-processing)
    vs = v * s
    vs_mean = vs.mean()
    buf = s + beta_u * vs_mean
    s = xm.all_reduce(xm.REDUCE_SUM, buf) * inv_W
    
    return s