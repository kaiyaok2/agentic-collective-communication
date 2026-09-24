
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.2 + 0.12*(r % 4) for r in range(W)]
    
    # Create coefficient tensor once
    if W > 1:
        b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).reshape(W, S)
    
    # 6 full iterations
    for _ in range(6):
        # Forward pass
        buf = s / W
        if W > 1:
            buf[:-1] = (s[:-1] + b_tensor * s[1:]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf.reshape(-1)).reshape(W, S)
        
        # Backward pass
        for r in range(W - 2, -1, -1):
            s[r] = s[r] - b[r] * s[r + 1]
    
    # Final forward pass
    buf = s / W
    if W > 1:
        buf[:-1] = (s[:-1] + b_tensor * s[1:]) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.reshape(-1))
