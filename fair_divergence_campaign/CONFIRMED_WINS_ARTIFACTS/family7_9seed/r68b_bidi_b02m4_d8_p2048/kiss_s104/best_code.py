
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b_list = [0.2 + 0.12*(r % 4) for r in range(W)]
    b_tensor = torch.tensor(b_list, device=x.device, dtype=x.dtype).unsqueeze(1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    for _ in range(6):
        # Forward pass (vectorized)
        buf = s / W
        buf[:-1] = (s[:-1] + b_tensor[:-1] * s[1:]) / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
        
        # Backward pass (must be sequential)
        for r in range(W - 2, -1, -1):
            s[r] = s[r] - b_tensor[r] * s[r + 1]
    
    # Final forward pass
    buf = s / W
    buf[:-1] = (s[:-1] + b_tensor[:-1] * s[1:]) / W
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
