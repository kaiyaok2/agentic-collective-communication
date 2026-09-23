def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 12
    OFF = 2
    
    # Compute count of ranks keeping each bucket
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(5):
            b = (st + j) % B
            c[b] += 1
    
    # Global sum across all ranks
    result = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create scale tensor
    scale = torch.ones_like(result)
    for b in range(B):
        scale[b*S:(b+1)*S] = c[b]
    
    # Single multiplication
    result = result * scale
    
    return result
