
def r43_route_B11_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 11; W = world_size; L = 3; OFF = 2; STR = 1
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # rank-indexed routing: THIS rank masks its own length-L window
    start = (rank + OFF) % B
    keep = [(start + STR*j) % B for j in range(L)]
    
    # Create mask and multiply
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    buf = s * mask
    
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    return result
