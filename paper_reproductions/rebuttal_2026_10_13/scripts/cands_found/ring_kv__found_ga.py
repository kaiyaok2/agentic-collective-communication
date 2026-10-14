def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    N_PARTS = 1
    results = []
    for chunk in kv_chunks:
        part_sz = chunk.numel() // N_PARTS
        parts = []
        for p in range(N_PARTS):
            piece = chunk[p * part_sz:(p + 1) * part_sz]
            gathered = xm.all_gather(piece.unsqueeze(0), dim=0)
            for p in range(N_PARTS):
                piece = chunk[p * part_sz:(p + 1) * part_sz]
                gathered = xm.all_gather(piece.unsqueeze(0), dim=0)
                parts.append(gathered.view(world_size, -1))
        results.append(torch.cat(parts, dim=1).reshape(-1))
    return results