
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    w_inv = 1.0 / W
    
    # Pre-compute scaled weights
    a_list = [1.0 + 0.6*(r % 4) for r in range(W)]
    inv_a_list = [1.0 / max(a, 1e-9) for a in a_list]
    
    # Create flattened weight tensors with scaling applied
    a_flat = []
    inv_a_flat = []
    for r in range(W):
        a_flat.extend([a_list[r] * w_inv] * S)
        inv_a_flat.extend([inv_a_list[r]] * S)
    
    a_tensor = torch.tensor(a_flat, device=x.device, dtype=x.dtype)
    inv_a_tensor = torch.tensor(inv_a_flat, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(7):
        s = s * a_tensor
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        if _ < 6:
            s = s * inv_a_tensor
    
    return s
