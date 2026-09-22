
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 11
    W = world_size
    L = 3
    OFF = 2
    STR = 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # This rank's window blocks
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    
    # Create mask tensor once (1.0 for kept blocks, 0.0 otherwise)
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep_blocks:
        mask[b*S:(b+1)*S] = 1.0
    
    # Create inverse overlap tensor for normalization once
    # Handle c[b] = 0 case by setting to 1.0 (no-op for normalization)
    inv_overlap = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if c[b] > 0:
            inv_overlap[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Perform 7 iterations
    for iteration in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            s = acc * inv_overlap
        else:
            s = acc
    
    return s
