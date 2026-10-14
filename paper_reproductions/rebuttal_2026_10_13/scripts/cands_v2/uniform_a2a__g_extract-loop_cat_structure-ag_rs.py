def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    transposed = gathered.view(world_size, world_size, chunk_size).permute(1, 0, 2).contiguous().view(-1)
    return xm.reduce_scatter(xm.REDUCE_SUM, transposed, scale=1.0 / world_size, scatter_dim=0, shard_count=world_size)
