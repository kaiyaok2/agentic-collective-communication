
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; W = world_size; L = 3; OFF = 2; STR = 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Use zeros_like instead of zeros
    mask = torch.zeros_like(s)
    start = (rank + OFF) % B
    for j in range(L):
        b = (start + STR * j) % B
        mask[b*S:(b+1)*S] = 1.0
    
    return xm.all_reduce(xm.REDUCE_SUM, s * mask)
