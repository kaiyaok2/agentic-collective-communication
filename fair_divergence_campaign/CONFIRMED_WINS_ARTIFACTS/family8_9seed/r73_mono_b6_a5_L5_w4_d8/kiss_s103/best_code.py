
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    SIG = [3, 2, 1, 0, 5, 4]
    
    # Pre-compute A array
    A = [0.0] * 6
    for r in range(W):
        w_r = 0.45 + 0.02 * (r % 4)
        st = (r + 1) % 6
        for j in range(5):
            A[(st + 1 * j) % 6] += w_r
    
    # Pre-compute which blocks to keep
    start = (rank + 1) % 6
    keep_list = [(start + j) % 6 for j in range(5)]
    
    # Pre-compute weight for this rank
    w = 0.45 + 0.02 * (rank % 4)
    
    # Create division weights as tensor with proper shape
    A_inv = torch.tensor([1.0 / a for a in A], dtype=x.dtype, device=x.device).view(6, 1)
    
    # Initial all_reduce and reshape to 2D
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(6, S)
    
    for iteration in range(7):
        buf = torch.zeros_like(s)
        
        # Fill buffer with weighted slices
        for b in keep_list:
            jb = SIG[b]
            buf[b] = w * s[jb]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(6, S)
        
        if iteration < 6:
            # Apply division and permute
            acc_scaled = acc * A_inv
            
            rec = torch.zeros_like(s)
            for b in range(6):
                jb = SIG[b]
                rec[jb] = acc_scaled[b]
            s = rec
        else:
            s = acc.view(-1)
    
    return s
