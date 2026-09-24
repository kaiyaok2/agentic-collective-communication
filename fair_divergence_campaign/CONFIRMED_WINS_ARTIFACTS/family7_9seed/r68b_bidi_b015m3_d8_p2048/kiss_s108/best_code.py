
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    inv_W = 1.0 / W
    
    # Precompute coupling tensor
    b_tensor = torch.tensor(b[:-1], dtype=x.dtype, device=x.device).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    for _ in range(5):
        # Forward sweep - vectorized
        buf = torch.zeros_like(s)
        buf[:W-1] = (s[:W-1] + b_tensor * s[1:W]) * inv_W
        buf[W-1] = s[W-1] * inv_W
        
        # Backward sweep - still needs to be sequential
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
        for r in range(W - 2, -1, -1):
            s[r] = s[r] - b[r] * s[r+1]
    
    # Final iteration
    buf = torch.zeros_like(s)
    buf[:W-1] = (s[:W-1] + b_tensor * s[1:W]) * inv_W
    buf[W-1] = s[W-1] * inv_W
    result = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return result
