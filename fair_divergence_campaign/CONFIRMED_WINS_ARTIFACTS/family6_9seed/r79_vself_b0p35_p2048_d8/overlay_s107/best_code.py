def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    
    # Precompute the fixed sign vector v
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(x.dtype)
    
    # Helper function for the forward step: s -> buf
    def forward_step(s):
        return s + BETA * v * (v * s).mean()
    
    # Helper function for the inverse step: acc -> s
    def inverse_step(acc):
        return acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # First all_reduce (initial sum)
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Pipeline: We'll overlap computation with communication
    # We use all_reduce_coalesced or issue async operations
    # Since xm.all_reduce is blocking by default, we need to structure
    # the code to minimize wait time by preparing the next buffer while
    # the communication is in flight.
    
    # For true pipelining, we'd need non-blocking collectives.
    # However, since the reference implementation shows sequential operations,
    # we'll structure the code to enable hardware-level pipelining by
    # organizing operations efficiently.
    
    # Iteration 1
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 2
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 3
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 4
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 5
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 6
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = inverse_step(acc)
    
    # Iteration 7 (final)
    buf = forward_step(s)
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / W
    s = acc
    
    return s