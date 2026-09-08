
def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: top-8 largest values across all ranks, sorted descending
    # Optimize masking: subtract large value from selected elements
    
    g = xm.all_gather(x, dim=0)
    
    vals = []
    
    # Iteration 1
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 2
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 3
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 4
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 5
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 6
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 7
    m = g.max()
    vals.append(m.unsqueeze(0))
    g = g - (g >= m).to(g.dtype) * 2e9
    
    # Iteration 8 (final, no masking needed)
    m = g.max()
    vals.append(m.unsqueeze(0))
    
    return torch.cat(vals, dim=0)
