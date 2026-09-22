
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute scaled coefficients
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_scaled = [(val / W) for val in a_list]
    a_inv_list = [1.0 / max(val, 1e-9) for val in a_list]
    
    # Create expanded tensors directly
    a_scaled_exp = torch.tensor([val for val in a_scaled for _ in range(S)],
                                device=x.device, dtype=x.dtype)
    a_inv_exp = torch.tensor([val for val in a_inv_list for _ in range(S)],
                             device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s = a_scaled_exp * s
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        s = s * a_inv_exp
    
    s = a_scaled_exp * s
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
