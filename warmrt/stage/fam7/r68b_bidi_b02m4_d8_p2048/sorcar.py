
def r68b_bidi_b02m4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(W, S)
    buf = s / W
    
    # Create coefficient tensor
    c_list = [(0.2 + 0.12 * (r % 4)) / W for r in range(W - 1)]
    c_tensor = torch.tensor(c_list, device=x.device, dtype=x.dtype).view(-1, 1)
    
    # Apply weighted addition
    buf[:-1] = buf[:-1] + c_tensor * s[1:]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf.view(-1))
