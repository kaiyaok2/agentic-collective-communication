
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b_list = [0.4 + 0.08*(r % 3) for r in range(W)]
    
    # Create b tensor for vectorized operations (W-1 values, each repeated S times)
    b_vec = torch.tensor(b_list[:-1], device=x.device, dtype=x.dtype).repeat_interleave(S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(6):
        # Forward pass - vectorized
        s_left = s[:-S]      # first W-1 chunks
        s_right = s[S:]      # last W-1 chunks (offset by S)
        buf_main = (s_left + b_vec * s_right) / W
        buf_last = s[-S:] / W
        buf = torch.cat([buf_main, buf_last], dim=0)
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass
        if iteration < 5:
            chunks = list(torch.chunk(s, W, dim=0))
            for i in range(W-2, -1, -1):
                chunks[i] = chunks[i] - b_list[i] * chunks[i+1]
            s = torch.cat(chunks, dim=0)
    
    return s
