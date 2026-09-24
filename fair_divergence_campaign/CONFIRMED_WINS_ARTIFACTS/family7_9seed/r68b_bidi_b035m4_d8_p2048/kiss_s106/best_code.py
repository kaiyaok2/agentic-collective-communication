
def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    inv_W = 1.0 / W
    b_vals = [0.35 + 0.1*(r % 4) for r in range(W)]
    
    # Create coefficient tensor
    b_fwd = torch.zeros(W * S, dtype=x.dtype, device=x.device)
    for r in range(W - 1):
        b_fwd[r*S:(r+1)*S] = b_vals[r]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Do 6 iterations
    for iteration in range(6):
        # Forward sweep - use narrow instead of cat+zeros
        s_shifted = torch.cat([s[S:], s[-S:] * 0.0])
        buf = (s + b_fwd * s_shifted) * inv_W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward sweep
        s_list = list(torch.chunk(s, W, dim=0))
        for r in range(W - 2, -1, -1):
            s_list[r] = s_list[r] - b_vals[r] * s_list[r+1]
        s = torch.cat(s_list)
    
    # Final forward sweep
    s_shifted = torch.cat([s[S:], s[-S:] * 0.0])
    buf = (s + b_fwd * s_shifted) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
