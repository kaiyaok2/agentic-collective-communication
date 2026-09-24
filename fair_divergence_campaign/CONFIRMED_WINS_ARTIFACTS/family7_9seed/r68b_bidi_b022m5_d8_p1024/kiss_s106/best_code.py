
def r68b_bidi_b022m5_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    b = [0.22 + 0.11*(r % 5) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x) / W
    
    b_mod = b[:-1] + [0.0]
    b_t = torch.tensor(b_mod, device=s.device, dtype=s.dtype).unsqueeze(1)
    sv = s.view(W, S)
    
    # Forward
    ss = torch.cat([sv[1:], sv[-1:]], dim=0)
    buf = sv + b_t * ss
    
    sv = xm.all_reduce(xm.REDUCE_SUM, buf.view(-1)).view(W, S)
    
    # Backward
    for r in range(W - 2, -1, -1):
        sv[r] -= b[r] * sv[r+1]
    
    # Final forward
    sv /= W
    ss = torch.cat([sv[1:], sv[-1:]], dim=0)
    buf = sv + b_t * ss
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
