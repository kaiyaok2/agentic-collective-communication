
def r68b_bidi_b015m3_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    result = s.clone()
    seg_size = S * (W - 1)
    left = s[:seg_size].view(W - 1, S)
    right = s[S:seg_size + S].view(W - 1, S)
    r_idx = torch.arange(W - 1, device=x.device, dtype=torch.long)
    b_coefs = (0.15 + 0.1 * (r_idx % 3)).to(x.dtype).unsqueeze(1)
    result[:seg_size] = (left + b_coefs * right).view(-1)
    return result
