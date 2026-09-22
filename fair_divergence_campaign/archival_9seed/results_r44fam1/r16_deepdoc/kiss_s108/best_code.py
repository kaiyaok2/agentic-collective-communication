def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Try building weights using torch.cat of repeated segments
    weights_list = []
    inv_weights_list = []
    
    for r in range(W):
        a = 1.0 + 0.5 * (r % 3)
        w_segment = torch.full((S,), a / W, device=x.device, dtype=x.dtype)
        iw_segment = torch.full((S,), 1.0 / a, device=x.device, dtype=x.dtype)
        weights_list.append(w_segment)
        inv_weights_list.append(iw_segment)
    
    weights_scaled = torch.cat(weights_list)
    inv_weights = torch.cat(inv_weights_list)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, weights_scaled * s) * inv_weights
    
    s = xm.all_reduce(xm.REDUCE_SUM, weights_scaled * s)
    
    return s