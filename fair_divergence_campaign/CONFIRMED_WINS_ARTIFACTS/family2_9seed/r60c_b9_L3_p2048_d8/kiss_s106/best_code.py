
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 9
    OFF = 2
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute which buckets this rank keeps
    start = (rank + OFF) % B
    b0 = start
    b1 = (start + 1) % B
    b2 = (start + 2) % B
    
    # Compute counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        c[st] += 1
        c[(st + 1) % B] += 1
        c[(st + 2) % B] += 1
    
    # Build divisor tensor efficiently
    div_parts = []
    for b in range(B):
        div_val = c[b] if c[b] > 0 else 1.0
        div_parts.append(torch.full((S,), div_val, device=x.device, dtype=x.dtype))
    div_tensor = torch.cat(div_parts, dim=0)
    
    # Perform 6 rounds with division
    for _ in range(6):
        buf = torch.zeros_like(s)
        buf[b0*S:(b0+1)*S] = s[b0*S:(b0+1)*S]
        buf[b1*S:(b1+1)*S] = s[b1*S:(b1+1)*S]
        buf[b2*S:(b2+1)*S] = s[b2*S:(b2+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / div_tensor
    
    # Final round without division
    buf = torch.zeros_like(s)
    buf[b0*S:(b0+1)*S] = s[b0*S:(b0+1)*S]
    buf[b1*S:(b1+1)*S] = s[b1*S:(b1+1)*S]
    buf[b2*S:(b2+1)*S] = s[b2*S:(b2+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
