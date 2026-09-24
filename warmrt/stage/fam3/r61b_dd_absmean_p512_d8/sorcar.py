def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # Keep tensor in [B, S] shape throughout most computation
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # First 6 iterations
    for _ in range(6):
        f = 1.0 + s.abs().mean(dim=1, keepdim=True)  # [B, 1] - no unsqueeze needed
        buf = s * f  # Broadcasting: [B, S] * [B, 1]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(B, S)
        s = acc / (world_size * f)
    
    # 7th iteration
    f = 1.0 + s.abs().mean(dim=1, keepdim=True)
    buf = s * f
    acc = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
    
    return acc / world_size
