
def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: top-8 largest values across all ranks concatenation, sorted descending.
    
    g = xm.all_gather(x, dim=0)
    
    # Unrolled loop for potential XLA optimization
    m0 = g.max(); g = g - (g >= m0) * 1e9
    m1 = g.max(); g = g - (g >= m1) * 1e9
    m2 = g.max(); g = g - (g >= m2) * 1e9
    m3 = g.max(); g = g - (g >= m3) * 1e9
    m4 = g.max(); g = g - (g >= m4) * 1e9
    m5 = g.max(); g = g - (g >= m5) * 1e9
    m6 = g.max(); g = g - (g >= m6) * 1e9
    m7 = g.max()
    
    return torch.stack([m0, m1, m2, m3, m4, m5, m6, m7])
