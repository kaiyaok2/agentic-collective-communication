
def r61b_dd_absmean_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # First 6 iterations
    for _ in range(6):
        s_blocks = s.view(B, S)
        abs_vals = s_blocks.abs()
        factors = [1.0 + abs_vals[b].mean() for b in range(B)]
        
        buf_blocks = [s_blocks[b] * factors[b] for b in range(B)]
        buf = torch.cat(buf_blocks, dim=0)
        
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        acc_blocks = acc.view(B, S)
        
        for b in range(B):
            acc_blocks[b] /= (world_size * factors[b])
        
        s = acc
    
    # Last iteration
    s_blocks = s.view(B, S)
    abs_vals = s_blocks.abs()
    factors = [1.0 + abs_vals[b].mean() for b in range(B)]
    
    buf_blocks = [s_blocks[b] * factors[b] for b in range(B)]
    buf = torch.cat(buf_blocks, dim=0)
    
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return acc / world_size
