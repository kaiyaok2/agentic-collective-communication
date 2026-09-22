def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 2
    
    # Compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create divisor tensor once
    c_tensor = torch.ones_like(x)
    for b in range(B):
        if c[b] > 0:
            c_tensor[b*S:(b+1)*S] = c[b]
    
    # This rank's blocks - create mask once
    start = (rank + OFF) % B
    mask = torch.zeros_like(x)
    for j in range(L):
        b = (start + STR*j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations using mask multiplication
    for iteration in range(7):
        buf = s * mask
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Divide by overlap count (skip on last iteration)
        if iteration < 6:
            s = s / c_tensor
    
    return s
