def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Create weight vector once (Python list -> tensor)
    a_list = [1.0 + 0.6*(r % 4) for r in range(W)]
    
    # Create weight tensors once
    weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    inv_weights = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    
    for r in range(W):
        weights[r*S:(r+1)*S] = a_list[r] / W
        inv_weights[r*S:(r+1)*S] = 1.0 / max(a_list[r], 1e-9)
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations of weight-reduce-unweight
    for _ in range(6):
        buf = s * weights
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_weights
    
    # Final weighted reduce
    buf = s * weights
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s