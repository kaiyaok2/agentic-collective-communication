
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    a = [1.0 + 0.4*(r % 5) for r in range(W)]
    
    # Create weight tensors once
    weight = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    inv_weight = torch.zeros(W * S, device=x.device, dtype=x.dtype)
    for r in range(W):
        weight[r*S:(r+1)*S] = a[r] / W
        inv_weight[r*S:(r+1)*S] = 1.0 / max(a[r], 1e-9)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Six iterations with inverse weight division
    for iteration in range(6):
        s = s * weight
        s = xm.all_reduce(xm.REDUCE_SUM, s)
        s = s * inv_weight
    
    # Final iteration without inverse weight division
    s = s * weight
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s
