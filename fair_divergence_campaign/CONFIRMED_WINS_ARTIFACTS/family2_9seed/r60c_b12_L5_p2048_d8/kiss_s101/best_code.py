
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    L = 5
    
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create normalization scale tensor once
    scale_vals = []
    for b in range(B):
        scale_vals.extend([1.0 / max(c[b], 1)] * S)
    scale = torch.tensor(scale_vals, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for i in range(7):
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if i < 6:
            acc = acc * scale
        
        s = acc
    
    return s
