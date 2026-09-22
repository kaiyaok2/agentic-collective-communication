
def r22_su8_count16_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Precompute weight tensors (vectorized)
    weight_list = []
    inv_weight_list = []
    for r in range(W):
        weight_list.extend([a[r] / W] * S)
        inv_weight_list.extend([1.0 / max(a[r], 1e-9)] * S)
    
    weights = torch.tensor(weight_list, device=x.device, dtype=x.dtype)
    inv_weights = torch.tensor(inv_weight_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations: weight, all_reduce, then unweight (except last)
    for i in range(6):
        buf = s * weights
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        if i < 5:
            s = s * inv_weights
    
    return s
