
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Window for this rank
    start = (rank + OFF) % B
    
    buf = torch.zeros_like(s)
    
    # Check if we can do a single contiguous copy
    if start + L <= B:
        # No wraparound - single slice
        buf[start*S:(start+L)*S] = s[start*S:(start+L)*S]
    else:
        # Wraparound - two slices
        blocks_before_wrap = B - start
        buf[start*S:B*S] = s[start*S:B*S]
        remaining = L - blocks_before_wrap
        buf[0:remaining*S] = s[0:remaining*S]
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
