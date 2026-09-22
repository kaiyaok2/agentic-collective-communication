
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    
    # Create coefficient tensor once - vectorized approach
    a_list = [1.0 + 0.25*(r % 5) for r in range(W)]
    a_tensor = torch.tensor(a_list, device=x.device, dtype=x.dtype)
    a_expanded = a_tensor.unsqueeze(1).repeat(1, S).view(-1)
    a_scaled = a_expanded / W
    a_clamped = torch.clamp(a_expanded, min=1e-9)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 6 full iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
        s = s / a_clamped
    
    # Final iteration
    s = xm.all_reduce(xm.REDUCE_SUM, a_scaled * s)
    
    return s
