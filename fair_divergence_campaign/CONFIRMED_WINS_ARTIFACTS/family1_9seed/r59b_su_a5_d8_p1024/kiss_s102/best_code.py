def r59b_su_a5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    
    # Efficient scale vector generation
    r_vals = torch.arange(W, device=x.device, dtype=torch.long)
    a_vals = 1.0 + 0.4 * (r_vals % 5).to(x.dtype)
    scale_vec = a_vals / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = s.view(W, S) * scale_vec.unsqueeze(1)
    s = s.view(-1)
    s = xm.all_reduce(xm.REDUCE_SUM, s)
    
    return s