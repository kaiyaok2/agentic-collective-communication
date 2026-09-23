def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.8
    dtype = x.dtype
    
    # Minimize collectives by doing more local work before synchronization
    # We'll do all 8 iterations but only synchronize strategically
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    buf = s / g
    acc = xm.all_reduce(xm.REDUCE_SUM, buf) / W
    
    # Do all remaining 7 iterations with a single collective at the end
    # Stack all intermediate buffers
    buffers = []
    
    for i in range(7):
        A = acc.abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s = acc * gr
        g = 1.0 + BETA * s.abs().mean()
        buf = s / g
        buffers.append(buf)
        # Use local buffer for next iteration
        acc = buf
    
    # Single stacked all_reduce for all 7 iterations
    stacked = torch.stack(buffers)
    reduced = xm.all_reduce(xm.REDUCE_SUM, stacked) / W
    
    # Return the final iteration result
    return reduced[-1]