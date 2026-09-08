
def evolved_p129(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y_r = concat(x_{(r+1) % W}, x_{(r+2) % W})
    g = xm.all_gather(x, dim=0).reshape(world_size, N)
    
    # Try to reduce operations by getting both at once
    idx = [(rank + 1) % world_size, (rank + 2) % world_size]
    selected = g[idx]
    
    return selected.reshape(-1)
