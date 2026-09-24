
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b_list = [0.15 + 0.1*(r % 3) for r in range(W)]
    inv_W = 1.0 / W
    
    # Pre-compute b as a tensor
    b_tensor = torch.tensor(b_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(7):
        # Forward pass
        chunks = list(torch.split(s, S))
        stacked = torch.stack(chunks)  # (W, S)
        
        # Compute new chunks: new[r] = (stacked[r] + b[r] * stacked[r+1]) * inv_W
        # Vectorize: new[0:W-1] = (stacked[0:W-1] + b[0:W-1] * stacked[1:W]) * inv_W
        curr = stacked[:-1]  # (W-1, S)
        next_ch = stacked[1:]  # (W-1, S)
        b_vec = b_tensor[:-1].unsqueeze(1)  # (W-1, 1)
        
        forward_result = (curr + b_vec * next_ch) * inv_W  # (W-1, S)
        last_chunk = stacked[-1:] * inv_W  # (1, S)
        
        combined = torch.cat([forward_result, last_chunk], dim=0)  # (W, S)
        buf = combined.reshape(-1)
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass
        if iteration < 6:
            chunks = list(torch.split(s, S))
            for r in range(W - 2, -1, -1):
                chunks[r] = chunks[r] - b_list[r] * chunks[r+1]
            s = torch.cat(chunks, dim=0)
    
    return s
