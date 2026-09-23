
def r68_bidi_b03_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    if W > 1:
        # Create coefficient tensor using arange for efficiency
        rank_ids = torch.arange(W - 1, device=x.device, dtype=torch.long)
        b_vals = 0.3 + 0.1 * (rank_ids % 4)
        b_tensor = b_vals.to(x.dtype).repeat_interleave(S)
        
        part1 = s[0:(W-1)*S]
        part2 = s[S:W*S]
        s[0:(W-1)*S] = part1 + b_tensor * part2
    
    return s
