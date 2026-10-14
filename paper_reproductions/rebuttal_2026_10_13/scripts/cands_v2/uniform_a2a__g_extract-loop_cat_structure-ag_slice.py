def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(input_tensor.unsqueeze(0), dim=0)
    flat = gathered.view(world_size, -1)
    return torch.cat([flat[src, rank * chunk_size:(rank + 1) * chunk_size] for src in range(world_size)], dim=0)
