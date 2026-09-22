
def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Simpler weight tensor creation
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    weights = []
    for val in a_list:
        weights.extend([val / W] * S)
    
    scale = torch.tensor(weights, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * scale)
    
    return s
