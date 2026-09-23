def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Pre-compute weight arrays in Python
    a_list = [1.0 + 0.4*(r % 5) for r in range(W)]
    weights_list = [a_list[r] / W for r in range(W)]
    inv_weights_list = [1.0 / max(a_list[r], 1e-9) for r in range(W)]
    
    # Create weight tensors
    weights = torch.tensor([[w] for w in weights_list], device=x.device, dtype=x.dtype)
    inv_weights = torch.tensor([[w] for w in inv_weights_list], device=x.device, dtype=x.dtype)
    
    # Initial all_reduce, keep in reshaped form
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    # 6 iterations of weighted averaging
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, (s * weights).view(W * S)).view(W, S) * inv_weights
    
    # Final weighted operation
    return xm.all_reduce(xm.REDUCE_SUM, (s * weights).view(W * S))