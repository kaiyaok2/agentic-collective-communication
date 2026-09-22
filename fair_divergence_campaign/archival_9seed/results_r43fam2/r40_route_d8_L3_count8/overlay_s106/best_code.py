def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 8
    W = world_size
    L = 3
    OFF = 2
    
    # Step 1: Initial all_reduce to get sum across all ranks
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute per-block overlap counts (how many ranks' windows cover each block)
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Determine which blocks this rank's window covers
    start = (rank + OFF) % B
    my_window = set((start + j) % B for j in range(L))
    
    # Pre-compute all 7 masked buffers locally
    # For iterations 1 through 7, we mask the window and prepare buffers
    # Iteration 7 is special (no division after), so we handle 6 iterations with division
    
    buffers = []
    current = s.clone()
    
    for iteration in range(7):
        # Mask: keep only blocks in my_window
        buf = torch.zeros_like(current)
        for b in range(B):
            if b in my_window:
                buf[b*S:(b+1)*S] = current[b*S:(b+1)*S]
        buffers.append(buf)
        
        # For iterations 0-5, we need to divide by c[b] after all_reduce
        # and use that as input for next iteration's masking
        # But we're batching, so we'll compute this after the batched all_reduce
        
        # Simulate what would happen: all_reduce, then divide
        if iteration < 6:
            # Simulate the all_reduce result (sum of masked buffers across ranks)
            # Then divide by c[b] for each block
            # This becomes the input for next iteration
            # But we can't actually do this here - we need to think differently
            
            # Actually, we need to prepare the buffers that WOULD be sent
            # in each iteration if we were doing them sequentially
            # After the batched all_reduce, we'll recover the intermediate results
            pass
    
    # Actually, let me reconsider the strategy more carefully
    # The reference does: mask -> all_reduce -> divide -> mask -> all_reduce -> divide -> ...
    # To batch, we need to prepare all masks based on intermediate states
    
    # Let's trace through what each iteration needs:
    # Iter 0: mask s -> all_reduce -> divide -> s'
    # Iter 1: mask s' -> all_reduce -> divide -> s''
    # ... etc
    
    # Since each iteration depends on the previous result, we can't truly batch them
    # unless we reformulate. However, the strategy says to pre-compute all 7 masked buffers.
    # This suggests we might be masking the SAME base tensor s for each iteration?
    
    # Re-reading: "Pre-compute all 7 masked buffers locally"
    # Perhaps the insight is that all ranks mask the same blocks (their window),
    # so we can prepare 7 copies of the same mask pattern, concatenate, do one all_reduce,
    # then process each chunk with the appropriate division from each iteration's c values?
    
    # Let me implement the simpler interpretation: just batch the communication
    # by concatenating the 7 buffers and doing one all_reduce
    
    # Create concatenated buffer (7 copies of masked s)
    concat_size = 7 * B * S
    concat_buf = torch.zeros(concat_size, dtype=x.dtype)
    
    current = s
    for iteration in range(7):
        # Mask current state
        for b in range(B):
            if b in my_window:
                offset = iteration * B * S + b * S
                concat_buf[offset:offset+S] = current[b*S:(b+1)*S]
        
        # Simulate division for next iteration (but we can't do this without all_reduce)
        # This approach won't work for dependent iterations
        
    # The batching strategy as described won't work for dependent operations.
    # Let me just implement it with reduced dispatches where possible.
    # Fall back to sequential execution:
    
    current = s
    for iteration in range(7):
        start = (rank + OFF) % B
        keep = set((start + j) % B for j in range(L))
        buf = torch.zeros_like(current)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = current[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        current = acc
    
    return current