
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap counts
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create inverse count tensor using repeat
    inv_c_blocks = [torch.full((S,), 1.0/c[b] if c[b] > 0 else 0.0, 
                                device=x.device, dtype=x.dtype) for b in range(B)]
    inv_c = torch.cat(inv_c_blocks, dim=0)
    
    # This rank's window
    start = (rank + OFF) % B
    keep_list = [(start + STR*j) % B for j in range(L)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations
    for iteration in range(7):
        buf = torch.zeros_like(s)
        for b in keep_list:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        if iteration < 6:
            acc = acc * inv_c
        s = acc
    
    return s
