
def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape
    s_reshaped = s.view(W, S)
    
    # Create coefficient tensor using arange for efficiency
    r_vals = torch.arange(W-1, device=x.device)
    b_tensor = (0.3 + 0.1 * (r_vals % 5)).to(x.dtype).view(-1, 1)
    
    # Compute transformation
    result = s_reshaped / W
    result[:-1] = (s_reshaped[:-1] + b_tensor * s_reshaped[1:]) / W
    
    # Final all_reduce
    result = xm.all_reduce(xm.REDUCE_SUM, result.reshape(-1))
    
    return result
