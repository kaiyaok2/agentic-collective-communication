def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; L = 3; OFF = 2
    
    # Get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create overlap count tensor
    overlap = torch.zeros_like(s)
    for b in range(B):
        overlap[b*S:(b+1)*S] = c[b]
    
    # Multiply by overlap counts
    s = s * overlap
    
    return s