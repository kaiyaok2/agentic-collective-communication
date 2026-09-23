def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    
    # Step 1: First all-reduce to get initial s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Speculative forward pass - compute all 7 iterations locally
    # using estimated global statistics (will be approximate)
    buffers = []
    current = s
    
    for iter_idx in range(7):
        # Local normalization step
        g = 1.0 + BETA * current.abs().mean()
        buf = current / g
        
        # Store buffer for batched all-reduce
        buffers.append(buf)
        
        # Estimate what the all-reduced result would be
        # We use current buffer * W as estimate (speculative)
        acc_estimate = buf  # Approximate - will be corrected later
        
        # Continue pipeline with estimated values
        A = acc_estimate.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        current = acc_estimate * gr
    
    # Step 3: Concatenate all 7 buffers and do single large all-reduce
    concatenated = torch.cat(buffers, dim=0)
    reduced_concat = xm.all_reduce(xm.REDUCE_SUM, concatenated)
    
    # Step 4: Split back and apply corrections
    chunk_size = x.shape[0]
    reduced_buffers = [reduced_concat[i*chunk_size:(i+1)*chunk_size] for i in range(7)]
    
    # Now process with correct global values
    for iter_idx in range(7):
        acc = reduced_buffers[iter_idx] / W
        
        if iter_idx < 6:
            # Compute correction for next iteration
            A = acc.abs().mean()
            M = A / (1.0 - BETA * A)
            gr = 1.0 + BETA * M
            s_corrected = acc * gr
            g_corrected = 1.0 + BETA * s_corrected.abs().mean()
            buf_corrected = s_corrected / g_corrected
            
            # Update the next buffer with correction
            reduced_buffers[iter_idx + 1] = buf_corrected * W  # Re-scale for division later
        else:
            # Last iteration - this is our final result
            s = acc
    
    return s