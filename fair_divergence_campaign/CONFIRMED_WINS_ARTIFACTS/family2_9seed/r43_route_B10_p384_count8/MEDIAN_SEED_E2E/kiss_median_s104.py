
def r43_route_B10_p384_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 10
    L = 3
    OFF = 2
    
    # Precompute overlap counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(L):
            c[(st + j) % B] += 1
    
    # Create divisor tensor using stack and repeat
    divisors_list = []
    for b in range(B):
        block_div = torch.full((S,), c[b], dtype=x.dtype, device=x.device)
        divisors_list.append(block_div)
    divisor = torch.cat(divisors_list, dim=0)
    
    # This rank's window blocks
    start = (rank + OFF) % B
    keep = [(start + j) % B for j in range(L)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # 7 iterations of mask, reduce, divide
    for it in range(7):
        buf = torch.zeros_like(s)
        for b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        if it < 6:
            s = s / divisor
    
    return s
