def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.3
    N = 16384
    dtype = x.dtype
    
    # Pre-compute the fixed sign vector v
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute all 8 projection buffers locally before any collective
    buffers = []
    for i in range(8):
        if i == 0:
            # First iteration uses the initial s
            buf = s + BETA * v * (v * s).mean()
        else:
            # Subsequent iterations: simulate the update s would have after all_reduce
            # We need to pre-compute what s would be after processing buf[i-1]
            # s_next = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
            # where acc = all_reduce(buf[i-1]) / W
            # Since we're pre-computing, we'll use the current s
            s_local = s
            buf = s_local + BETA * v * (v * s_local).mean()
        buffers.append(buf)
    
    # Issue 8 back-to-back all_reduce calls on pre-allocated tensors
    reduced_buffers = []
    for buf in buffers:
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        reduced_buffers.append(acc)
    
    # Post-process results sequentially
    s = s  # Start with initial s
    for i in range(8):
        acc = reduced_buffers[i] / W
        if i < 7:
            # Update s for next iteration
            s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        else:
            # Last iteration: just keep acc as s
            s = acc
    
    return s