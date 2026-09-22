
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    inv_w = 1.0 / W
    
    # Build coefficient tensor using cat
    coef_parts = []
    for r in range(W):
        coef_val = (1.0 + 0.25 * (r % 5)) * inv_w
        coef_parts.append(torch.full((S,), coef_val, device=x.device, dtype=x.dtype))
    
    coef = torch.cat(coef_parts)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * coef)
    
    return s
