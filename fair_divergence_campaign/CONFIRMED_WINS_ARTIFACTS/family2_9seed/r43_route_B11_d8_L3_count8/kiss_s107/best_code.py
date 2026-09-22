
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Pre-compute overlap counts
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # Compute this rank's window indices
    start = (rank + OFF) % B
    window_blocks = [(start + STR*j) % B for j in range(L)]
    
    # Create single mask tensor with division factors embedded
    mask_with_div = torch.zeros(B * S, device=x.device, dtype=x.dtype)
    for b in window_blocks:
        if c[b] > 0:
            mask_with_div[b*S:(b+1)*S] = 1.0 / c[b]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations with division
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, s * mask_with_div)
    
    # For final iteration, create mask on the fly by multiplying back by c
    final_mask = torch.zeros_like(mask_with_div)
    for b in window_blocks:
        if c[b] > 0:
            final_mask[b*S:(b+1)*S] = mask_with_div[b*S:(b+1)*S] * c[b]
    
    s = xm.all_reduce(xm.REDUCE_SUM, s * final_mask)
    
    return s
