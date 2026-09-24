
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    b = [0.4 + 0.08*(r % 3) for r in range(W)]
    b_tensor = torch.tensor([b[r] if r < W-1 else 0.0 for r in range(W)], 
                             device=x.device, dtype=x.dtype)
    
    s_reshaped = s.view(W, S)
    s_shifted = torch.cat([s_reshaped[1:], torch.zeros(1, S, device=x.device, dtype=x.dtype)], dim=0)
    
    blended = ((s_reshaped + b_tensor.view(W, 1) * s_shifted) / W).view(-1)
    
    return xm.all_reduce(xm.REDUCE_SUM, blended)
