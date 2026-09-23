def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute weights as Python list
    a_list = [1.0 + 0.6*(r % 4) for r in range(W)]
    a_inv_list = [1.0 / max(a_list[r], 1e-9) for r in range(W)]
    
    # Create and expand weight tensors
    a = torch.tensor(a_list, device=x.device, dtype=x.dtype)
    a_inv = torch.tensor(a_inv_list, device=x.device, dtype=x.dtype)
    
    # Precompute expanded weights with W division folded in
    a_div_W = a.repeat_interleave(S) / W
    a_inv_expanded = a_inv.repeat_interleave(S)
    
    # First all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with full cycle
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, a_div_W * s)
        s = s * a_inv_expanded
    
    # Final iteration: apply and reduce only
    s = xm.all_reduce(xm.REDUCE_SUM, a_div_W * s)
    
    return s