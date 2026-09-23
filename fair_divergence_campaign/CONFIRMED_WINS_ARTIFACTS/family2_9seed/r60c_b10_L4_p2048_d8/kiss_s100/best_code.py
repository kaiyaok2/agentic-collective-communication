
def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    
    # Compute which buckets this rank keeps
    start = (rank + OFF) % B
    keep = [(start + 1*j) % B for j in range(4)]
    
    # Compute contributor counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(4):
            c[(st + 1*j) % B] += 1
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single iteration: mask and aggregate
    buf = torch.zeros_like(s)
    for b in keep:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
