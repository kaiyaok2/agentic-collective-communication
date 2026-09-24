
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.2 + 0.12*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create coefficient tensor
    coeff = torch.zeros_like(s)
    for r in range(W - 1):
        coeff[r*S:(r+1)*S] = b[r]
    
    # Create shifted version (shift by S elements)
    s_shifted = torch.cat([s[S:], torch.zeros(S, device=s.device, dtype=s.dtype)])
    
    # Vectorized computation
    buf = (s + coeff * s_shifted) / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
