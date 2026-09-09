
def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: top-8 largest values across all ranks, sorted descending
    # Fully unrolled loop
    
    g = xm.all_gather(x, dim=0)
    big = 1e9
    
    # Iteration 1
    m0 = g.max()
    g = g - (g >= m0).to(g.dtype) * big
    
    # Iteration 2
    m1 = g.max()
    g = g - (g >= m1).to(g.dtype) * big
    
    # Iteration 3
    m2 = g.max()
    g = g - (g >= m2).to(g.dtype) * big
    
    # Iteration 4
    m3 = g.max()
    g = g - (g >= m3).to(g.dtype) * big
    
    # Iteration 5
    m4 = g.max()
    g = g - (g >= m4).to(g.dtype) * big
    
    # Iteration 6
    m5 = g.max()
    g = g - (g >= m5).to(g.dtype) * big
    
    # Iteration 7
    m6 = g.max()
    g = g - (g >= m6).to(g.dtype) * big
    
    # Iteration 8
    m7 = g.max()
    
    return torch.stack([m0, m1, m2, m3, m4, m5, m6, m7], dim=0)
