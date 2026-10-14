def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    return [xm.all_gather(c.unsqueeze(0), dim=0).view(-1) for c in kv_chunks]
