def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Compute scaling factors for each rank as a tensor
    a = torch.tensor([1.0 + 0.25 * (r % 5) for r in range(W)], 
                     dtype=dtype, device=x.device)
    
    # Single all-reduce to get the global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to enable vectorized operations across shards
    # Shape: (W, S) where W is world_size and S is shard size
    s_reshaped = s.view(W, S)
    
    # Create scaling vector: shape (W, 1) for broadcasting
    a_vec = a.view(W, 1)
    
    # Simulate 7 iterations of dependent all-reduces using vectorized operations
    for iteration in range(7):
        # Apply scaling: buf = a[r] * s / W for each shard r
        # Shape: (W, S)
        buf = a_vec * s_reshaped / W
        
        # Simulate all-reduce: multiply by W to get the effect of summing W copies
        s_new = buf * W
        
        # Apply inverse scaling: divide by a[r] for each shard
        s_reshaped = s_new / a_vec.clamp(min=1e-9)
    
    # Final iteration (8th): scale by a[r] / W and sum
    buf = a_vec * s_reshaped / W
    
    # Simulate final all-reduce
    s_final = buf * W
    
    # Reshape back to flat vector
    return s_final.view(-1)