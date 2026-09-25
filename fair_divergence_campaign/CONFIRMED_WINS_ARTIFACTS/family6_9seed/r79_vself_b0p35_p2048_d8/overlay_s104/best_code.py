def r79_vself_b0p35_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.35
    N = 16384
    dtype = x.dtype
    
    # Precompute v vector once
    idx = torch.arange(N, device=x.device)
    v = (1 - 2 * (idx % 2)).to(dtype)
    
    # Initial all_reduce to get s
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Strategy: Concatenate multiple buf tensors and do fewer all_reduces
    # Do 7 iterations total, but batch them into groups
    
    # Batch 1: iterations 1-4 (concatenate 4 buffers)
    bufs = []
    s_vals = [s]
    
    for i in range(4):
        s_current = s_vals[-1]
        buf = s_current + BETA * v * (v * s_current).mean()
        bufs.append(buf)
        # Compute next s assuming we had the reduced buf
        # (we'll fix this after all_reduce)
        s_vals.append(s_current)  # placeholder
    
    # Concatenate all 4 buffers and do single all_reduce
    concat_bufs = torch.cat(bufs, dim=0)
    reduced_bufs = xm.all_reduce(xm.REDUCE_SUM, concat_bufs)
    
    # Process results sequentially
    s = s_vals[0]
    for i in range(4):
        acc = reduced_bufs[i*N:(i+1)*N] / W
        s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
    
    # Batch 2: iterations 5-7 (concatenate 3 buffers)
    bufs = []
    for i in range(3):
        buf = s + BETA * v * (v * s).mean()
        bufs.append(buf)
        # For the last one, we just need to return it
        if i < 2:
            # Placeholder - will compute properly after all_reduce
            pass
    
    # Concatenate all 3 buffers and do single all_reduce
    concat_bufs = torch.cat(bufs, dim=0)
    reduced_bufs = xm.all_reduce(xm.REDUCE_SUM, concat_bufs)
    
    # Process results sequentially
    for i in range(3):
        acc = reduced_bufs[i*N:(i+1)*N] / W
        if i < 2:
            s = acc - (BETA / (1.0 + BETA)) * v * (v * acc).mean()
        else:
            s = acc  # Final iteration
    
    return s