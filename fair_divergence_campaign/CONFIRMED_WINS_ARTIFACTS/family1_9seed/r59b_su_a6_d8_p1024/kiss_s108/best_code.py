
def r59b_su_a6_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Try using cat to build weights
    weight_chunks = []
    for r in range(W):
        w = (1.0 + 0.35 * (r % 6)) / W
        chunk = torch.full((S,), w, device=s.device, dtype=s.dtype)
        weight_chunks.append(chunk)
    w_tensor = torch.cat(weight_chunks)
    return xm.all_reduce(xm.REDUCE_SUM, s * w_tensor)
