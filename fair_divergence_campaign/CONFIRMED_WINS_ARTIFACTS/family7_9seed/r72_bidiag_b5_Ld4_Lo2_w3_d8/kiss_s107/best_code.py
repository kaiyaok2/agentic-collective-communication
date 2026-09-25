
def r72_bidiag_b5_Ld4_Lo2_w3_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    
    s_flat = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute A and C
    A = [0.0] * 5
    C = [0.0] * 5
    for r in range(world_size):
        wd = 0.9 + 0.04 * (r % 3)
        wo = 0.25 + 0.01 * (r % 3)
        for j in range(4):
            A[(r % 5 + j) % 5] += wd
        for j in range(2):
            b = ((r + 1) % 5 + j) % 5
            if b >= 1:
                C[b] += wo
    
    wd = 0.9 + 0.04 * (rank % 3)
    wo = 0.25 + 0.01 * (rank % 3)
    kd = [((rank % 5) + j) % 5 for j in range(4)]
    ko = [(((rank + 1) % 5) + j) % 5 for j in range(2) if (((rank + 1) % 5) + j) % 5 >= 1]
    
    # Create weight mask
    wd_mask = torch.zeros(5, 1, dtype=s_flat.dtype, device=s_flat.device)
    for b in kd:
        wd_mask[b, 0] = wd
    
    # Work in reshaped space
    s = s_flat.view(5, S)
    
    for _ in range(6):
        buf = wd_mask * s
        for b in ko:
            buf[b] += wo * s[b-1]
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(5, S)
        s[0] /= A[0]
        for b in range(1, 5):
            s[b] = (s[b] - C[b] * s[b-1]) / A[b]
    
    # Final iteration
    buf = wd_mask * s
    for b in ko:
        buf[b] += wo * s[b-1]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
