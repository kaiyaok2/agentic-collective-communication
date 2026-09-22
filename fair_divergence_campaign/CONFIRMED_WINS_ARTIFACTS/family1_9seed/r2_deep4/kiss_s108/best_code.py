def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create weight vectors and expand
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Build full tensors more efficiently
    a_w_list = []
    a_inv_list = []
    for r in range(W):
        a_w_list.extend([a[r] / W] * S)
        a_inv_list.extend([1.0 / max(a[r], 1e-9)] * S)
    
    a_w = torch.tensor(a_w_list, device=x.device, dtype=x.dtype)
    a_inv = torch.tensor(a_inv_list, device=x.device, dtype=x.dtype)
    
    # Iterations
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = a_inv * xm.all_reduce(xm.REDUCE_SUM, a_w * s)
    s = a_inv * xm.all_reduce(xm.REDUCE_SUM, a_w * s)
    s = xm.all_reduce(xm.REDUCE_SUM, a_w * s)
    
    return s