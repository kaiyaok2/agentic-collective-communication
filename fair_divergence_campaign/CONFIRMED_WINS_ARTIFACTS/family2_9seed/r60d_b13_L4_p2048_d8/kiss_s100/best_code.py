
def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute counts once
    c = [0] * 13
    for r in range(world_size):
        st = (r + 2) % 13
        for j in range(4):
            c[(st + j) % 13] += 1
    
    # Compute keep set
    start = (rank + 2) % 13
    keep_set = {(start + j) % 13 for j in range(4)}
    
    # Create mask and count divisor tensors once
    mask = torch.zeros_like(s)
    count_div = torch.ones_like(s)
    for b in range(13):
        if b in keep_set:
            mask[b*S:(b+1)*S] = 1.0
        if c[b] > 0:
            count_div[b*S:(b+1)*S] = 1.0 / c[b]
    
    # 6 full iterations with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc * count_div
    
    # Final iteration without division
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
