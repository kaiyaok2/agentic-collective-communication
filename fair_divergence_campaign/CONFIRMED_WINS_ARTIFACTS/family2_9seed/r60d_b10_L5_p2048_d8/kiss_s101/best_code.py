
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pre-compute bucket counts
    c = [0] * 10
    for r in range(W):
        st = (r + 2) % 10
        for j in range(5):
            c[(st + j) % 10] += 1
    
    # Build scale tensor with fewer operations
    scale_list = []
    for b in range(10):
        scale_val = 1.0 / c[b] if c[b] > 0 else 1.0
        scale_list.append(torch.full((S,), scale_val, device=x.device, dtype=x.dtype))
    scale = torch.cat(scale_list, dim=0)
    
    # My keep pattern
    start = (rank + 2) % 10
    
    # 7 iterations
    for iteration in range(7):
        buf = torch.zeros_like(s)
        if start + 5 <= 10:
            buf[start*S:(start+5)*S] = s[start*S:(start+5)*S]
        else:
            buf[start*S:10*S] = s[start*S:10*S]
            end = (start + 5) % 10
            buf[0:end*S] = s[0:end*S]
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            acc = acc * scale
        s = acc
    
    return s
