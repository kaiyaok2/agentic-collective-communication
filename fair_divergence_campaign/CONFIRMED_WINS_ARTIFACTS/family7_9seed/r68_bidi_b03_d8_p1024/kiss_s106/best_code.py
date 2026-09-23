
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.3 + 0.1 * (r % 5) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with backward pass
    for _ in range(6):
        # Split into chunks
        chunks = [s[r*S:(r+1)*S] for r in range(W)]
        
        # Forward pass
        new_chunks = []
        for r in range(W - 1):
            new_chunks.append((chunks[r] + b[r] * chunks[r+1]) / W)
        new_chunks.append(chunks[W-1] / W)
        
        buf = torch.cat(new_chunks, dim=0)
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass - need to update in place
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r] * s[(r+1)*S:(r+2)*S]
    
    # Last iteration without backward pass
    chunks = [s[r*S:(r+1)*S] for r in range(W)]
    new_chunks = []
    for r in range(W - 1):
        new_chunks.append((chunks[r] + b[r] * chunks[r+1]) / W)
    new_chunks.append(chunks[W-1] / W)
    buf = torch.cat(new_chunks, dim=0)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
