def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for c in kv_chunks:
        n = c.numel() // 8
        parts = [xm.all_gather(c[i * n:(i + 1) * n].unsqueeze(0), dim=0) for i in range(8)]
        outs.append(torch.cat([p.view(world_size, -1) for p in parts], dim=1).reshape(-1))
    return outs
