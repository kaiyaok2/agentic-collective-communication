def evolved_fused_reducescatter(tensors, rank, world_size, num_devices,
                                cores_per_device, xm, torch, num_nodes=1):
    """Zeros + all_gather + sum + split. Zero local ops."""
    sizes = [t.numel() for t in tensors]
    total = sum(sizes)
    flat = torch.zeros(1, total, device=tensors[0].device, dtype=tensors[0].dtype)
    offset = 0
    for t, sz in zip(tensors, sizes):
        flat[0, offset:offset + sz] = t
        offset += sz
    gathered = xm.all_gather(flat, dim=0)
    summed = gathered.sum(dim=0)
    results = []
    offset = 0
    for sz in sizes:
        shard_sz = sz // world_size
        start = offset + rank * shard_sz
        results.append(summed[start:start + shard_sz])
        offset += sz
    return results
