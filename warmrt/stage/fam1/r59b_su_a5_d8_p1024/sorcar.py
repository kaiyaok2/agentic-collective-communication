
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Precompute weight tensors
    a_list = [1.0 + 0.4*(r % 5) for r in range(W)]
    a_inv_list = [1.0 / max(a, 1e-9) for a in a_list]
    
    # Create repeated weight vectors, pre-scaled by 1/W
    a_vec_scaled = torch.cat([torch.full((S,), a_list[r]/W, device=x.device, dtype=x.dtype) for r in range(W)])
    a_inv_vec = torch.cat([torch.full((S,), a_inv_list[r], device=x.device, dtype=x.dtype) for r in range(W)])
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 full iterations with inverse weights
    for _ in range(5):
        s = xm.all_reduce(xm.REDUCE_SUM, a_vec_scaled * s)
        s = s * a_inv_vec
    
    # Final iteration without inverse weights
    s = xm.all_reduce(xm.REDUCE_SUM, a_vec_scaled * s)
    
    return s
