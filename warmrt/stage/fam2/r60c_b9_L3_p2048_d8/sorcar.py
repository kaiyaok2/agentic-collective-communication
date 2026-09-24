
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 9
    OFF = 2
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep = [(start + j) % B for j in range(3)]
    
    # Compute bucket counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(3):
            c[(st + j) % B] += 1
    
    # Build mask
    mask_pattern = [1.0 if b in keep else 0.0 for b in range(B)]
    mask = torch.tensor([val for val in mask_pattern for _ in range(S)], 
                        device=x.device, dtype=x.dtype)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final round without normalization
    s = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
