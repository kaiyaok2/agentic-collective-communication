
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try to create b_tensor without loop
    # Create indices for each chunk: 0, 1, 2, 3, ... repeated S times each
    chunk_indices = torch.arange(W * S, device=x.device, dtype=torch.long) // S
    # Compute b values: 0.4 + 0.08 * (chunk_index % 3)
    b_tensor = 0.4 + 0.08 * (chunk_indices % 3).to(x.dtype)
    # Zero out the last chunk
    b_tensor[(W-1)*S:] = 0.0
    
    # Shift s - reuse part of s
    s_shifted = torch.cat([s[S:], s[:S]], dim=0)
    
    # Vectorized computation
    buf = (s + b_tensor * s_shifted) / W
    
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
