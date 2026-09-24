
def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 13
    OFF = 2
    
    # Compute counts once
    c = [0] * B
    for r in range(world_size):
        start = (r + OFF) % B
        for j in range(4):
            c[(start + j) % B] += 1
    
    # Determine which buckets this rank keeps
    start = (rank + OFF) % B
    keep_set = set((start + j) % B for j in range(4))
    
    # Create mask
    mask = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in keep_set:
        mask[b*S:(b+1)*S] = 1.0
    
    # Initial all_reduce + final round without scaling
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
