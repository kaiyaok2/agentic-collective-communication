
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.4 + 0.08*(r % 3) for r in range(W)]
    
    # Create coefficient tensor
    b_tensor = torch.tensor(b[:W-1], device=x.device, dtype=x.dtype).unsqueeze(1).expand(W-1, S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    # 6 full forward-backward iterations
    for _ in range(6):
        # Forward pass - vectorized
        buf = s / W
        buf[:W-1] = (s[:W-1] + b_tensor * s[1:W]) / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
        
        # Backward pass - must be sequential due to dependencies
        for r in range(W - 2, -1, -1):
            s[r] -= b[r] * s[r+1]
    
    # Final forward pass
    buf = s / W
    buf[:W-1] = (s[:W-1] + b_tensor * s[1:W]) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
