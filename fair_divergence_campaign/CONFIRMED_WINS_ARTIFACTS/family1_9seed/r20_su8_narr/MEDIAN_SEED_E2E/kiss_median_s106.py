
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute weights in Python
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_scaled = [a[r] / W for r in range(W)]
    a_inv = [1.0/max(a[r], 1e-9) for r in range(W)]
    
    # Create weight vectors directly
    a_vec_scaled = torch.tensor([val for val in a_scaled for _ in range(S)], dtype=x.dtype, device=x.device)
    a_inv_vec = torch.tensor([val for val in a_inv for _ in range(S)], dtype=x.dtype, device=x.device)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 iterations with normalization
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, s * a_vec_scaled) * a_inv_vec
    
    # Final iteration without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, s * a_vec_scaled)
    
    return s
