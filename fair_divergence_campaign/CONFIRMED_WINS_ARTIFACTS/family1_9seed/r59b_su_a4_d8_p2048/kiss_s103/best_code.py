def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """Optimized with minimal tensor operations."""
    S = 2048
    W = world_size
    
    # Create base weight vectors
    a = [1.0 + 0.6 * (r % 4) for r in range(W)]
    
    # Build weight tensors by stacking
    w_list = [torch.full((S,), v / W, device=x.device, dtype=x.dtype) for v in a]
    w = torch.cat(w_list, dim=0)
    
    w_inv_list = [torch.full((S,), 1.0 / max(v, 1e-9), device=x.device, dtype=x.dtype) for v in a]
    w_inv = torch.cat(w_inv_list, dim=0)
    
    # Process
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, w * s) * w_inv
    s = xm.all_reduce(xm.REDUCE_SUM, w * s)
    
    return s