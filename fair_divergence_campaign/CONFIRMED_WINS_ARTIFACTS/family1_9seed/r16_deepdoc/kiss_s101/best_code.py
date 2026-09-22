
def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Precompute weights
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Create weight tensors using cat
    w_fwd_parts = []
    w_inv_parts = []
    for r in range(W):
        w = a[r] / W
        inv = 1.0 / max(a[r], 1e-9)
        w_fwd_parts.append(torch.full((S,), w, device=x.device, dtype=x.dtype))
        w_inv_parts.append(torch.full((S,), inv, device=x.device, dtype=x.dtype))
    
    w_fwd = torch.cat(w_fwd_parts)
    w_inv = torch.cat(w_inv_parts)
    
    # Computation
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, w_fwd * s) * w_inv
    
    return xm.all_reduce(xm.REDUCE_SUM, w_fwd * s)
