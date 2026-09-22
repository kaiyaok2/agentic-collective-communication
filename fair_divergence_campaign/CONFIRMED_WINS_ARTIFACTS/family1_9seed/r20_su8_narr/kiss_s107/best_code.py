
def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute scaling vectors directly as flat lists
    a_list = [1.0 + 0.5*(r % 3) for r in range(W)]
    scale_fwd_list = []
    scale_bwd_list = []
    for r in range(W):
        scale_fwd_list.extend([a_list[r] / W] * S)
        scale_bwd_list.extend([1.0 / max(a_list[r], 1e-9)] * S)
    
    scale_fwd = torch.tensor(scale_fwd_list, device=x.device, dtype=x.dtype)
    scale_bwd = torch.tensor(scale_bwd_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 full iterations with unscale
    for _ in range(5):
        buf = scale_fwd * s
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = scale_bwd * s
    
    # Final iteration without unscale
    buf = scale_fwd * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
