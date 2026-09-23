
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Create weight tensors
    b_pattern = [0.3, 0.4, 0.5, 0.6]
    b_list = [b_pattern[r % 4] for r in range(W-1)]
    
    inv_W = 1.0 / W
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Reshape for easier manipulation: [W, S]
    s_2d = s.view(W, S)
    
    # Create b_tensor as 1D for broadcasting
    b_1d = torch.tensor(b_list, device=x.device, dtype=x.dtype).view(-1, 1)
    
    # First forward + all_reduce
    buf_2d = s_2d * inv_W
    buf_2d[:-1] = (s_2d[:-1] + b_1d * s_2d[1:]) * inv_W
    buf = buf_2d.view(-1)
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_2d = s.view(W, S)
    
    # 5 iterations
    for _ in range(5):
        # Backward sweep - still needs loop due to dependencies
        for r in range(W - 2, -1, -1):
            s_2d[r] = s_2d[r] - b_list[r] * s_2d[r + 1]
        # Forward sweep
        buf_2d = s_2d * inv_W
        buf_2d[:-1] = (s_2d[:-1] + b_1d * s_2d[1:]) * inv_W
        buf = buf_2d.view(-1)
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s_2d = s.view(W, S)
    
    return s.view(-1)
