
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.4 + 0.08 * (r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor for forward pass vectorization
    b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).repeat_interleave(S)
    W_inv = 1.0 / W
    
    # 6 iterations with forward and backward
    for _ in range(6):
        # Forward pass - vectorized, allocate once
        buf = s * W_inv
        buf[:(W-1)*S] = (s[:(W-1)*S] + b_tensor * s[S:W*S]) * W_inv
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass
        for r in range(W - 2, -1, -1):
            idx = r * S
            s[idx:idx+S] = s[idx:idx+S] - b[r] * s[idx+S:idx+2*S]
    
    # Final forward pass
    buf = s * W_inv
    buf[:(W-1)*S] = (s[:(W-1)*S] + b_tensor * s[S:W*S]) * W_inv
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
