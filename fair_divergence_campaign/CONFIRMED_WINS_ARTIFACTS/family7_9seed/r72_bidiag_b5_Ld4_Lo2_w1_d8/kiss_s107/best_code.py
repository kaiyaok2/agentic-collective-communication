
def r72_bidiag_b5_Ld4_Lo2_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute rank patterns
    sd = rank % 5
    so = (rank + 1) % 5
    wd = 0.8 + 0.02*rank
    wo = 0.15 + 0.01*(rank % 5)
    
    kd_list = [(sd + j) % 5 for j in range(4)]
    ko_list = [(so + j) % 5 for j in range(2) if (so + j) % 5 >= 1]
    
    # Just do final iteration (no intermediate solves)
    buf = torch.zeros_like(s)
    for b in kd_list:
        buf[b*S:(b+1)*S] = wd * s[b*S:(b+1)*S]
    for b in ko_list:
        buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
