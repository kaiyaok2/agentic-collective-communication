
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b_list = [0.15 + 0.1*(r % 3) for r in range(W)]
    
    # Reshape to work with chunks
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    
    # Create coefficient tensor
    b = torch.tensor(b_list[:-1] + [0.0], device=x.device, dtype=x.dtype).view(W, 1)
    
    # Initial forward
    s_next = torch.cat([s[1:], torch.zeros(1, S, device=x.device, dtype=x.dtype)], dim=0)
    buf = (s + b * s_next) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
    
    # 6 iterations
    for _ in range(6):
        # Backward sweep
        s_next = torch.cat([s[1:], torch.zeros(1, S, device=x.device, dtype=x.dtype)], dim=0)
        s = s - b * s_next
        
        # Forward sweep
        buf = (s + b * s_next) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
    
    return s.view(-1)
