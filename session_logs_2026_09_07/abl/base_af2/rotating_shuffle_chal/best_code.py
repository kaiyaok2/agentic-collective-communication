def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y_r = concat(x_{(r+1) % W}, x_{(r+2) % W})
    
    g = xm.all_gather(x, dim=0).view(world_size, N)
    
    # Compute indices as Python ints
    idx1 = (rank + 1) % world_size  
    idx2 = (rank + 2) % world_size
    
    # Try to get both at once with a list
    neighbors = g[[idx1, idx2]]  # Should give (2, N) tensor
    
    # Flatten
    return neighbors.view(-1)