
def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Build weight arrays
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_over_w = []
    inv_a = []
    for r in range(W):
        a_over_w.extend([a[r] / W] * S)
        inv_a.extend([1.0 / max(a[r], 1e-9)] * S)
    
    a_over_w_t = torch.tensor(a_over_w, device=x.device, dtype=x.dtype)
    inv_a_t = torch.tensor(inv_a, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, a_over_w_t * s) * inv_a_t
    
    # Last iteration
    s = xm.all_reduce(xm.REDUCE_SUM, a_over_w_t * s)
    
    return s
