
def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    L = 5
    
    # Pre-compute counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Pre-compute keep set for this rank
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(L))
    
    # Create mask tensor (1 for keep, 0 for discard)
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        if b in keep:
            mask[b*S:(b+1)*S] = 1.0
    
    # Create count tensor for division
    c_tensor = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        c_tensor[b*S:(b+1)*S] = c[b] if c[b] > 0 else 1.0
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations
    for it in range(7):
        buf = s * mask
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Last iteration doesn't divide
        if it < 6:
            s = acc / c_tensor
        else:
            s = acc
    
    return s
