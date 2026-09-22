
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Build weights using cat
    weight_chunks = []
    for r in range(W):
        weight_chunks.append(torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype))
    weights = torch.cat(weight_chunks)
    
    return xm.all_reduce(xm.REDUCE_SUM, s * weights)
