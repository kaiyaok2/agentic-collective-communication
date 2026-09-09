def evolved_grad_ar(rep_grads, rank, world_size, num_devices,
                    cores_per_device, xm, torch, num_nodes=1):
    if not rep_grads:
        return []
    inv = 1.0 / world_size
    chunk_bytes = 64 * 1024 * 1024
    cur_idx, cur_bytes = [], 0
    buckets = []
    for i, g in enumerate(rep_grads):
        b = g.numel() * g.element_size()
        if cur_idx and cur_bytes + b > chunk_bytes:
            buckets.append(cur_idx)
            cur_idx, cur_bytes = [], 0
        cur_idx.append(i)
        cur_bytes += b
    if cur_idx:
        buckets.append(cur_idx)
    out = [None] * len(rep_grads)
    for idxs in buckets:
        if len(idxs) == 1:
            i = idxs[0]
            out[i] = xm.all_reduce(xm.REDUCE_SUM, rep_grads[i]) * inv
            continue
        shapes = [rep_grads[i].shape for i in idxs]
        flat = torch.cat([rep_grads[i].reshape(-1) for i in idxs])
        flat = xm.all_reduce(xm.REDUCE_SUM, flat) * inv
        offs = 0
        for i, shp in zip(idxs, shapes):
            n = 1
            for d in shp:
                n = n * d
            out[i] = flat[offs:offs+n].reshape(shp)
            offs += n
    return out
