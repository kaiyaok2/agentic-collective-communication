
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Pre-compute weight tensors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    w_fwd = []
    w_inv = []
    for r in range(W):
        w_fwd.extend([a[r] / W] * S)
        w_inv.extend([1.0 / max(a[r], 1e-9)] * S)
    
    weights_fwd = torch.tensor(w_fwd, device=x.device, dtype=x.dtype)
    weights_inv = torch.tensor(w_inv, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 5 complete iterations
    for _ in range(5):
        buf = s * weights_fwd
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * weights_inv
    
    # Final iteration without division
    buf = s * weights_fwd
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
