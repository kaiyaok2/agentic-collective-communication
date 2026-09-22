
def r43_route_B10_strided_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 10; W = world_size; L = 3; OFF = 2; STR = 2
    
    # Precompute keep blocks
    start = (rank + OFF) % B
    keep_list = [(start + STR*j) % B for j in range(L)]
    
    # Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Single iteration: mask and reduce (no division)
    buf = torch.zeros_like(s)
    for b in keep_list:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return result
