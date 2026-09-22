
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; L = 3; OFF = 2
    start = (rank + OFF) % B
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Clone and zero out non-window blocks
    buf = s.clone()
    for b in range(B):
        if b < start or b >= start + L:
            if not (start + L > B and b < (start + L) % B):
                buf[b*S:(b+1)*S] = 0
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
