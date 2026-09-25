
def r71_wdiag_b5_L3_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    start = rank % 5
    keep = [(start + j) % 5 for j in range(3)]
    w = 0.5 + 0.02 * rank
    
    # Try using multiplication with a mask
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = w
    
    buf = mask * s
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
