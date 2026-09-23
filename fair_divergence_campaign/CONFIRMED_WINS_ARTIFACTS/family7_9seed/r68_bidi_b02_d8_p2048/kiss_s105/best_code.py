
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = torch.tensor([0.2 + 0.1*(r % 3) for r in range(W)], dtype=x.dtype, device=x.device)
    b_col = b[:-1].unsqueeze(1)
    
    # Initial all_reduce - after this, all ranks have identical data
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    # All iterations are local since all ranks have identical data
    for _ in range(6):
        # Forward sweep (note: dividing by W then multiplying by W cancels out)
        buf = s.clone()
        buf[:-1] = s[:-1] + b_col * s[1:]
        s = buf
        
        # Backward sweep
        for r in range(W - 2, -1, -1):
            s[r] -= b[r] * s[r+1]
    
    # Final forward sweep
    buf = s.clone()
    buf[:-1] = s[:-1] + b_col * s[1:]
    
    return buf.reshape(-1)
