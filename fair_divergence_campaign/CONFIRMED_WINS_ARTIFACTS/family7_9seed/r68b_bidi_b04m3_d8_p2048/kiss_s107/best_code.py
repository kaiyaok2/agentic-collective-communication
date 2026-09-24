
def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    b = [0.4 + 0.08*(r % 3) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create b tensor for vectorized ops
    b_tensor = torch.tensor(b, device=x.device, dtype=x.dtype).view(W, 1)
    
    # 6 full iterations
    for _ in range(6):
        # Forward pass - vectorized
        s_view = s.view(W, S)
        buf = torch.zeros_like(s)
        buf_view = buf.view(W, S)
        buf_view[:-1] = (s_view[:-1] + b_tensor[:-1] * s_view[1:]) / W
        buf_view[-1] = s_view[-1] / W
        
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Backward pass - still needs loop due to dependencies
        s_view = s.view(W, S)
        result = torch.zeros_like(s)
        result_view = result.view(W, S)
        result_view[W-1] = s_view[W-1]
        for r in range(W - 2, -1, -1):
            result_view[r] = s_view[r] - b[r] * result_view[r+1]
        s = result
    
    # Final forward pass
    s_view = s.view(W, S)
    buf = torch.zeros_like(s)
    buf_view = buf.view(W, S)
    buf_view[:-1] = (s_view[:-1] + b_tensor[:-1] * s_view[1:]) / W
    buf_view[-1] = s_view[-1] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
