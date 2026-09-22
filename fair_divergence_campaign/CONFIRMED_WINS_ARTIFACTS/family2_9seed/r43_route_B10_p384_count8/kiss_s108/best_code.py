
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 10; W = world_size; L = 3; OFF = 2; STR = 1
    
    # Compute overlap count for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + STR*j) % B] += 1
    
    # All_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Scale each block by its overlap count
    result = torch.zeros_like(s)
    for b in range(B):
        result[b*S:(b+1)*S] = s[b*S:(b+1)*S] * c[b]
    
    return result
