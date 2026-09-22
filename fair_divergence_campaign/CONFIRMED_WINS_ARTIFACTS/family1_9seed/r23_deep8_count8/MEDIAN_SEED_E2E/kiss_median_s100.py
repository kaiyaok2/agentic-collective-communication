def r23_deep8_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Create coefficient tensors using torch operations
    weight_parts = [torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype) for r in range(W)]
    weight_coeff = torch.cat(weight_parts)
    
    unweight_parts = [torch.full((S,), 1.0 / max(a[r], 1e-9), device=x.device, dtype=x.dtype) for r in range(W)]
    unweight_coeff = torch.cat(unweight_parts)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * weight_coeff)
        s = s * unweight_coeff
    
    s = xm.all_reduce(xm.REDUCE_SUM, s * weight_coeff)
    
    return s