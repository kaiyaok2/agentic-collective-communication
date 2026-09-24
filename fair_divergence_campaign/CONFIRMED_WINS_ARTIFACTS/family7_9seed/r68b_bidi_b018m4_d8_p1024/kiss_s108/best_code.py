
def r68b_bidi_b018m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # All-reduce and scale
    s = xm.all_reduce(xm.REDUCE_SUM, x) / W
    
    # Try to vectorize the transformation
    # Create index tensors for batch processing
    chunks = torch.split(s, S)
    
    # Stack adjacent chunks for vectorized operation
    curr_chunks = torch.stack([chunks[r] for r in range(W - 1)])
    next_chunks = torch.stack([chunks[r + 1] for r in range(W - 1)])
    b_tensor = torch.tensor([0.18 + 0.13 * (r % 4) for r in range(W - 1)], 
                            device=x.device, dtype=x.dtype).unsqueeze(1)
    
    # Vectorized computation
    transformed = curr_chunks + b_tensor * next_chunks
    
    # Copy back
    for r in range(W - 1):
        s[r*S:(r+1)*S] = transformed[r]
    
    # Second all_reduce
    return xm.all_reduce(xm.REDUCE_SUM, s)
