
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 10
    W = world_size
    L = 3
    OFF = 2
    STR = 2
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute overlap counts as Python list
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR * j) % B] += 1
    
    # Determine which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = set((start + STR * j) % B for j in range(L))
    
    # Build mask list directly
    mask_list = []
    for b in range(B):
        if b in keep_blocks:
            mask_list.extend([1.0] * S)
        else:
            mask_list.extend([0.0] * S)
    mask = torch.tensor(mask_list, device=x.device, dtype=x.dtype)
    
    # Build divisor list directly
    divisor_list = []
    for b in range(B):
        divisor_list.extend([max(c[b], 1)] * S)
    divisor = torch.tensor(divisor_list, device=x.device, dtype=x.dtype)
    
    # 6 iterations with division
    for _ in range(6):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = acc / divisor
    
    # Final iteration without division  
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
