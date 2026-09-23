
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    
    # Compute counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(4):
            c[(st + j) % B] += 1
    
    # Create count tensor for vectorized division using reshape/expand
    c_tensor = torch.tensor(c, device=x.device, dtype=x.dtype)
    c_expanded = c_tensor.view(B, 1).expand(B, S).reshape(-1)
    
    # Which buckets to keep
    start = (rank + OFF) % B
    keep_list = [(start + j) % B for j in range(4)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations
    for it in range(7):
        buf = torch.zeros_like(s)
        for b in keep_list:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        if it < 6:
            acc = acc / c_expanded
        s = acc
    
    return s
