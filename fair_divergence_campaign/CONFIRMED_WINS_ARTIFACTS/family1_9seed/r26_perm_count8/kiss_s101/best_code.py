
def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create scales using torch operations
    half_shift = W // 2
    p_range = torch.arange(W, device=x.device, dtype=torch.long)
    inv_perm_vals = (p_range + half_shift) % W
    a_vals = 1.0 + 0.5 * (inv_perm_vals % 3).to(x.dtype)
    scales = a_vals / W
    
    return xm.all_reduce(xm.REDUCE_SUM, (s.view(W, S) * scales.unsqueeze(1)).view(-1))
