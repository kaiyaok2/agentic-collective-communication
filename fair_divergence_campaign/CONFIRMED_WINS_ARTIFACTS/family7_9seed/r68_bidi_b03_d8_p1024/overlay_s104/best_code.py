def r68_bidi_b03_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    dtype = x.dtype
    
    # Precompute coupling coefficients
    b = [0.3 + 0.1*(r % 5) for r in range(W)]
    
    # Strategy: Reduce collectives by doing 2 stages per all_reduce
    # This cuts collectives from 8 to 4
    
    def backward_sweep(s, b_coeffs):
        """Apply backward sweep in-place"""
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b_coeffs[r]*s[(r+1)*S:(r+2)*S]
        return s
    
    def forward_sweep(s, b_coeffs):
        """Apply forward sweep, return new buffer"""
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b_coeffs[r]*s[(r+1)*S:(r+2)*S]) / W
        return buf
    
    # Initial all_reduce + stage 1-2 processing
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = forward_sweep(s, b)
    s2 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s2, b)
    buf = forward_sweep(s2, b)
    
    # Stage 3-4
    s3 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s3, b)
    buf = forward_sweep(s3, b)
    s4 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s4, b)
    buf = forward_sweep(s4, b)
    
    # Stage 5-6
    s5 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s5, b)
    buf = forward_sweep(s5, b)
    s6 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s6, b)
    buf = forward_sweep(s6, b)
    
    # Stage 7-8
    s7 = xm.all_reduce(xm.REDUCE_SUM, buf)
    backward_sweep(s7, b)
    buf = forward_sweep(s7, b)
    s8 = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s8