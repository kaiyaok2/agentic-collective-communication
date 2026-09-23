
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = torch.tensor([0.3 + 0.1*(r % 4) for r in range(W-1)], device=x.device, dtype=x.dtype)
    b_expanded = b.unsqueeze(-1).expand(-1, S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        # Forward sweep
        s_reshaped = s.view(W, S)
        chunks_curr = s_reshaped[:-1]
        chunks_next = s_reshaped[1:]
        new_chunks = (chunks_curr + b_expanded * chunks_next) / W
        last_chunk = s_reshaped[-1:] / W
        s = xm.all_reduce(xm.REDUCE_SUM, torch.cat([new_chunks, last_chunk], dim=0).view(-1))
        
        # Backward sweep - use reshape too
        s_reshaped = s.view(W, S)
        for r in range(W - 2, -1, -1):
            s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
        s = s_reshaped.view(-1)
    
    # Final iteration
    s_reshaped = s.view(W, S)
    chunks_curr = s_reshaped[:-1]
    chunks_next = s_reshaped[1:]
    new_chunks = (chunks_curr + b_expanded * chunks_next) / W
    last_chunk = s_reshaped[-1:] / W
    s = xm.all_reduce(xm.REDUCE_SUM, torch.cat([new_chunks, last_chunk], dim=0).view(-1))
    
    return s
