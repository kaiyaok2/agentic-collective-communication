
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.22 + 0.11*(r % 5) for r in range(W)]
    W_inv = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(6):
        # Forward pass - use torch.split
        s_chunks = list(torch.split(s[:W*S], S))
        buf_chunks = [(s_chunks[r] + b[r] * s_chunks[r+1]) * W_inv for r in range(W - 1)]
        buf_chunks.append(s_chunks[-1] * W_inv)
        
        if s.numel() > W * S:
            buf_chunks.append(s[W*S:] * W_inv)
        
        buf = torch.cat(buf_chunks)
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass (skip on last iteration)
        if iteration < 5:
            s_chunks = list(torch.split(s[:W*S], S))
            for r in range(W - 2, -1, -1):
                s[r*S:(r+1)*S] = s_chunks[r] - b[r] * s_chunks[r+1]
    
    return s
