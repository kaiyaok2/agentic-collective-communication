
def r40_route_d6_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # per-block overlap count
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Build mask for this rank's window
    start = (rank + OFF) % B
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for j in range(L):
        b = (start + j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Build inverse overlap counts tensor (avoid div by zero)
    inv_c = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            inv_c[b*S:(b+1)*S] = 1.0 / c[b]
    
    for iteration in range(5):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 4:
            s = acc * inv_c
        else:
            s = acc
    
    return s
