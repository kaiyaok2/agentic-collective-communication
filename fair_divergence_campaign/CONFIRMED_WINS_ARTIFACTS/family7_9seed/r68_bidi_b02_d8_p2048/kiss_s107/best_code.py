
def r68_bidi_b02_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.2 + 0.1*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
    s = s_reshaped.view(-1)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
    s = s_reshaped.view(-1)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
    s = s_reshaped.view(-1)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
    s = s_reshaped.view(-1)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    s_reshaped = s.view(W, S)
    for r in range(W - 2, -1, -1):
        s_reshaped[r] = s_reshaped[r] - b[r] * s_reshaped[r+1]
    s = s_reshaped.view(-1)
    buf = s / W
    s_reshaped = s.view(W, S)
    buf_reshaped = buf.view(W, S)
    for r in range(W - 1):
        buf_reshaped[r] = (s_reshaped[r] + b[r] * s_reshaped[r+1]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf_reshaped.view(-1))
    
    return s
