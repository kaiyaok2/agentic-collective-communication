
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute weight tensors
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    weights = []
    inv_weights = []
    for r in range(W):
        weights.extend([a_list[r] / W] * S)
        inv_weights.extend([1.0 / max(a_list[r], 1e-9)] * S)
    weight_tensor = torch.tensor(weights, device=x.device, dtype=x.dtype)
    inv_weight_tensor = torch.tensor(inv_weights, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 iterations with vectorized operations
    for _ in range(5):
        buf = weight_tensor * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_weight_tensor
    
    # Final iteration without inverse weighting
    buf = weight_tensor * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
