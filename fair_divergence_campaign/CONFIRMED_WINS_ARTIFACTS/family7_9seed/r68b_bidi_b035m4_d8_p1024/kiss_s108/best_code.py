
def r68b_bidi_b035m4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024
    W = world_size
    b = [0.35 + 0.09*(r % 4) for r in range(W)]
    
    # Precompute tensors with shapes ready to use
    b_tensor = torch.tensor(b[:-1], device=x.device, dtype=x.dtype).unsqueeze(1)
    inv_W = 1.0 / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Vectorized forward sweep
    def forward_sweep(s):
        current = s[:(W-1)*S].reshape(W-1, S)
        next_part = s[S:W*S].reshape(W-1, S)
        transformed = ((current + b_tensor * next_part) * inv_W).reshape(-1)
        last = s[(W-1)*S:W*S] * inv_W
        return torch.cat([transformed, last], dim=0)
    
    # Backward sweep using split
    def backward_sweep(s):
        chunks = list(torch.split(s, S))
        for r in range(W - 2, -1, -1):
            chunks[r] = chunks[r] - b[r] * chunks[r+1]
        return torch.cat(chunks, dim=0)
    
    # 6 iterations
    for _ in range(6):
        s = xm.all_reduce(xm.REDUCE_SUM, forward_sweep(s))
        s = backward_sweep(s)
    
    # Final forward and all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, forward_sweep(s))
    
    return s
