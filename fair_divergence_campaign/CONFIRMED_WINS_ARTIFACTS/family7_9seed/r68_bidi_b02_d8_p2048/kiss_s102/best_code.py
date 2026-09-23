
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s_view = s.view(W, S)
    
    # Compute coefficients using arange
    r_idx = torch.arange(W, device=x.device, dtype=x.dtype)
    b = (0.2 + 0.1 * (r_idx % 3)).unsqueeze(1)
    
    # Create result
    result = s / W
    result_view = result.view(W, S)
    result_view[:-1] = (s_view[:-1] + b[:-1] * s_view[1:]) / W
    
    output = xm.all_reduce(xm.REDUCE_SUM, result)
    
    return output
