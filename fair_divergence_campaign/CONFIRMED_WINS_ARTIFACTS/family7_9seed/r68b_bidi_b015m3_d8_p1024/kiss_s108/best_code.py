
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    
    # Create coefficient tensor for vectorization
    b_list = [0.15 + 0.1*(r % 3) for r in range(W-1)]
    b_tensor = torch.tensor(b_list, device=x.device, dtype=x.dtype).repeat_interleave(S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s / W
    
    # Vectorized forward pass
    buf[0:(W-1)*S] = (s[0:(W-1)*S] + b_tensor * s[S:W*S]) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
