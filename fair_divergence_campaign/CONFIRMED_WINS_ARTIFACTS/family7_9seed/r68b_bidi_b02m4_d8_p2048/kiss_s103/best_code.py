
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape to [W, S]
    s_reshaped = s.view(W, S)
    
    # Create coupling coefficients
    r_indices = torch.arange(W-1, device=x.device, dtype=x.dtype)
    b_tensor = (0.2 + 0.12 * (r_indices % 4)).view(W-1, 1)
    
    # Vectorized coupling (no division since we'll avoid multiply at end)
    coupled = s_reshaped[:-1] + b_tensor * s_reshaped[1:]
    buf_reshaped = torch.cat([coupled, s_reshaped[-1:]], dim=0)
    
    return buf_reshaped.view(-1)
