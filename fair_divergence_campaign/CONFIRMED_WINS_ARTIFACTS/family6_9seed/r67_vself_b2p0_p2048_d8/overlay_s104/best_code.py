def r67_vself_b2p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 2.0
    N = 16384
    
    # Pre-compute constants
    inv_W = 1.0 / W
    beta_factor = BETA / (1.0 + BETA)
    
    # Create the fixed pattern vector v (alternating +1, -1)
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Initial all-reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Unroll iterations and fuse operations
    for _ in range(6):
        # Compute v dot s efficiently
        v_dot_s = (v * s).mean()
        # Compute buffer with fused operations
        buf = s + (BETA * v_dot_s) * v
        # All-reduce
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        # Apply scaling and correction in one step
        acc *= inv_W
        s = acc - (beta_factor * (v * acc).mean()) * v
    
    # Final iteration (no correction needed)
    v_dot_s = (v * s).mean()
    buf = s + (BETA * v_dot_s) * v
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc * inv_W
    
    return s