
def r43_route_B12_d8_L3_res_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 12; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # This rank's window blocks
    start = (rank + OFF) % B
    keep = [(start + STR * j) % B for j in range(L)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try 1 iteration
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
