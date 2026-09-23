
def r60c_b8_L2_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 8; W = world_size; L = 2; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # This rank's blocks
    start = (rank + OFF) % B
    b0 = start
    b1 = (start + 1) % B
    
    # 1 full iteration
    buf = torch.zeros_like(s)
    buf[b0*S:(b0+1)*S] = s[b0*S:(b0+1)*S]
    buf[b1*S:(b1+1)*S] = s[b1*S:(b1+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        if c[b] > 0:
            acc[b*S:(b+1)*S] /= c[b]
    s = acc
    
    # Final iteration without division
    buf = torch.zeros_like(s)
    buf[b0*S:(b0+1)*S] = s[b0*S:(b0+1)*S]
    buf[b1*S:(b1+1)*S] = s[b1*S:(b1+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
