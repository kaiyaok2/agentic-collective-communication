
def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    B = 5
    
    # Precompute which buckets this rank keeps
    start = rank % B
    keep = [(start + j) % B for j in range(2)]
    
    # Precompute counts for each bucket
    c = [0] * B
    for r in range(world_size):
        st = r % B
        for j in range(2):
            c[(st + j) % B] += 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create mask tensor
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Single masked all_reduce
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
