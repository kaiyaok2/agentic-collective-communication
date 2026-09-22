
def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Build weights using repeat
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    weight_fwd = torch.cat([torch.full((S,), a[r] / W, device=x.device, dtype=x.dtype) for r in range(W)])
    weight_inv = torch.cat([torch.full((S,), 1.0 / max(a[r], 1e-9), device=x.device, dtype=x.dtype) for r in range(W)])
    
    # Apply iterations
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * weight_fwd) * weight_inv
    s = xm.all_reduce(xm.REDUCE_SUM, s * weight_fwd) * weight_inv
    s = xm.all_reduce(xm.REDUCE_SUM, s * weight_fwd)
    
    return s
