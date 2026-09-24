
def r68b_bidi_b035m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute scaled coefficients
    b = [0.35 + 0.1*(r % 4) for r in range(W)]
    b_tensor = torch.zeros((W-1) * S, device=x.device, dtype=x.dtype)
    for r in range(W - 1):
        b_tensor[r*S:(r+1)*S] = b[r] / W
    
    inv_W = 1.0 / W
    
    for _ in range(6):
        # Forward pass with precomputed scaling
        buf = s * inv_W
        buf[:(W-1)*S] += b_tensor * s[S:]
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass
        for r in range(W - 2, -1, -1):
            idx = r * S
            s[idx:idx+S] -= b[r] * s[idx+S:idx+2*S]
    
    # Final forward pass
    buf = s * inv_W
    buf[:(W-1)*S] += b_tensor * s[S:]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
