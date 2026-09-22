
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; W = world_size; L = 3; OFF = 2; STR = 1
    
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Mask using zeros and slicing
    buf = torch.zeros_like(s)
    for b in keep_blocks:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
