
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.3 + 0.1*(r % 5) for r in range(W)]
    
    # Pre-compute b_tensor for forward sweeps
    b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).view(-1, 1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for iteration in range(6):
        # Forward sweep - vectorized
        s_view = s.view(W, S)
        buf = torch.cat([(s_view[:-1] + b_tensor * s_view[1:]).view(-1) / W, s_view[-1] / W], dim=0)
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward sweep - try using narrow for efficiency
        s_view = s.view(W, S)
        for r in range(W - 2, -1, -1):
            s_view[r] = s_view[r] - b[r] * s_view[r+1]
    
    # Final forward sweep
    s_view = s.view(W, S)
    buf = torch.cat([(s_view[:-1] + b_tensor * s_view[1:]).view(-1) / W, s_view[-1] / W], dim=0)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
