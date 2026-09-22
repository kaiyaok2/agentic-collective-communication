
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; W = world_size; L = 3; OFF = 2; STR = 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute overlap counts once
    c = [0]*B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Compute this rank's keep set once
    start = (rank + OFF) % B
    keep = set((start + STR*j) % B for j in range(L))
    
    # Create normalization tensor once (only for blocks that need it)
    norm_list = []
    for b in range(B):
        norm_list.extend([1.0/c[b] if c[b] > 0 else 1.0] * S)
    norm = torch.tensor(norm_list, device=x.device, dtype=x.dtype)
    
    # Helper to create masked buffer
    def mask_buffer(tensor):
        buf = torch.zeros_like(tensor)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = tensor[b*S:(b+1)*S]
        return buf
    
    # 6 rounds of mask -> all_reduce -> normalize
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, mask_buffer(s)) * norm
    
    # Final round without normalization
    s = xm.all_reduce(xm.REDUCE_SUM, mask_buffer(s))
    
    return s
