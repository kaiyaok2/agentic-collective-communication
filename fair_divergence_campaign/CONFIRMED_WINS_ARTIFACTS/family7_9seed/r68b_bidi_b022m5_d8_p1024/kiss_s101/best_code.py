
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    buf = s / W
    if W > 1:
        # Build b_tensor more efficiently
        b_list = [0.22 + 0.11*(r % 5) for r in range(W-1)]
        b_expanded = []
        for b_val in b_list:
            b_expanded.extend([b_val] * S)
        b_tensor = torch.tensor(b_expanded, device=x.device, dtype=x.dtype)
        
        # Vectorized computation
        part1 = s[:(W-1)*S]
        part2 = s[S:W*S]
        buf[:(W-1)*S] = (part1 + b_tensor * part2) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
