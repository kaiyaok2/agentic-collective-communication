def r22_su8_count16_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Try using reduce_scatter instead
    shard = xm.reduce_scatter(xm.REDUCE_SUM, x, scale=1.0, scatter_dim=0, shard_count=W)
    
    # Each rank has 1/W of the data, scale it
    s = shard * a[rank]
    
    # Gather back
    result = xm.all_gather(s.unsqueeze(0), dim=0).view(-1)
    
    return result
