
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    
    # Pre-compute overlap counts and create division tensor
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Create division factor tensor
    div_factor = torch.ones(B * S, device=x.device, dtype=x.dtype)
    for b in range(B):
        div_factor[b*S:(b+1)*S] = c[b]
    
    # Gather initial data from all ranks
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    s = gathered.sum(dim=0)
    
    # Pre-compute which blocks this rank keeps
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    
    # 7 iterations
    for iteration in range(7):
        buf = torch.zeros_like(s)
        for b in keep_blocks:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if iteration < 6:
            s = acc / div_factor
        else:
            s = acc
    
    return s
