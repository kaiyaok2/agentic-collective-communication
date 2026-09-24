
def r68b_bidi_b015m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.15 + 0.1*(r % 3) for r in range(W)]
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s * inv_W  # Initialize buf as s/W
    buf_view = buf.view(W, S)
    s_view = s.view(W, S)
    
    # Iteration 1
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_view = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_view[r] = s_view[r] - b[r]*s_view[r+1]
    
    # Iteration 2
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_view = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_view[r] = s_view[r] - b[r]*s_view[r+1]
    
    # Iteration 3
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_view = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_view[r] = s_view[r] - b[r]*s_view[r+1]
    
    # Iteration 4
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_view = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_view[r] = s_view[r] - b[r]*s_view[r+1]
    
    # Iteration 5
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    s_view = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_view[r] = s_view[r] - b[r]*s_view[r+1]
    
    # Iteration 6
    for r in range(W - 1):
        buf_view[r] = (s_view[r] + b[r]*s_view[r+1]) * inv_W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
