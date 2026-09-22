
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    scale = 1.0 / W
    
    # Construct vectors directly without repeat_interleave
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    scaled_a_vec = torch.tensor([a * scale for a in a_list for _ in range(S)], 
                                device=x.device, dtype=x.dtype)
    inv_a_vec = torch.tensor([1.0/max(a, 1e-9) for a in a_list for _ in range(S)], 
                             device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s = scaled_a_vec * s
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        s = s * inv_a_vec
    
    s = scaled_a_vec * s
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
