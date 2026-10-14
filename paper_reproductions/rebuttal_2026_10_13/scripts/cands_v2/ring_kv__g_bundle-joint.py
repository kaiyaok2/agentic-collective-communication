def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    sizes = [c.numel() for c in kv_chunks]
    flat = torch.cat(kv_chunks, dim=0)
    g = xm.all_gather(flat.unsqueeze(0), dim=0)
    outs = []; off = 0
    for sz in sizes:
        outs.append(g[:, off:off + sz].reshape(-1)); off += sz
    return outs
