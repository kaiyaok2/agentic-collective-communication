def r67_vself_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 1.0
    N = 16384
    depth = 8
    
    dtype = x.dtype
    
    # Compute initial s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create v pattern
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Pre-compute all 8 "buf" vectors using approximate s values
    # We'll maintain an estimate of s and compute all buffers at once
    buffers = []
    s_estimate = s.clone()
    
    for i in range(depth):
        # Compute buf_i = s_estimate + BETA * v * (v * s_estimate).mean()
        buf_i = s_estimate + BETA * v * (v * s_estimate).mean()
        buffers.append(buf_i)
        
        # Update s_estimate for next iteration using approximation
        # We approximate what acc would be after all-reduce and averaging
        # acc ≈ buf_i (assuming all ranks have similar values)
        acc_approx = buf_i
        # Apply inverse transform
        s_estimate = acc_approx - (BETA / (1.0 + BETA)) * v * (v * acc_approx).mean()
    
    # Pack all buffers into a single tensor [depth, N]
    packed_buffers = torch.stack(buffers, dim=0)  # [8, 16384]
    
    # Flatten for all-reduce
    packed_flat = packed_buffers.reshape(-1)  # [8*16384]
    
    # Single all-reduce dispatch
    all_reduced_flat = xm.all_reduce(xm.REDUCE_SUM, packed_flat)
    
    # Unpack and average
    all_reduced = all_reduced_flat.reshape(depth, N)  # [8, 16384]
    all_reduced = all_reduced / W
    
    # Apply inverse transforms sequentially to get final s
    s_result = s  # Start from initial s
    for i in range(depth):
        acc = all_reduced[i]
        if i < depth - 1:
            # Apply inverse transform
            s_result = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        else:
            # Last iteration: s = acc directly
            s_result = acc
    
    return s_result