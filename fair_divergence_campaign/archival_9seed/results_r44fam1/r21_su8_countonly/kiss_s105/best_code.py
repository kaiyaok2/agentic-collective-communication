
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Try using torch.cat for weight tensor creation
    weight_chunks = []
    for r in range(W):
        w = (1.0 + 0.5*(r % 3)) / W
        weight_chunks.append(torch.full((S,), w, device=x.device, dtype=x.dtype))
    weight_tensor = torch.cat(weight_chunks)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s * weight_tensor
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
