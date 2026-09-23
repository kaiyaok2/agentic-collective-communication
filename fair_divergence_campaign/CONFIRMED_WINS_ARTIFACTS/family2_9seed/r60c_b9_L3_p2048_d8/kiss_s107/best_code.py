
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 9
    OFF = 2
    
    # Pre-compute keep indices
    start = (rank + 2) % B
    keep_list = [(start + 1*j) % B for j in range(3)]
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final selective pass
    buf = torch.zeros_like(s)
    for b in keep_list:
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
