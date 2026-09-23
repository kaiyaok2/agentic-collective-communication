
def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Compute coefficients
    a = [1.0 + 0.4*(r % 5) for r in range(W)]
    
    # Create only forward weights, compute inverse on the fly
    fwd_w = torch.cat([torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype) for r in range(W)])
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations with computed inverse
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * fwd_w)
        # Compute inverse weight tensor on the fly
        inv_w = torch.cat([torch.full((S,), 1.0 / max(a[r], 1e-9), device=x.device, dtype=x.dtype) for r in range(W)])
        s = s * inv_w
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, s * fwd_w)
    
    return s
