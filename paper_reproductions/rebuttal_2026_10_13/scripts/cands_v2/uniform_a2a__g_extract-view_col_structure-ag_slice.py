def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    return gathered.view(world_size, world_size, chunk_size)[:, rank, :].contiguous().view(-1)
