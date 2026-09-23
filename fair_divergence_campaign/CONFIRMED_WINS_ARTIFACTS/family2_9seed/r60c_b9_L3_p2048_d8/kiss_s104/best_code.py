
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 9
    OFF = 2
    
    all_sum = xm.all_reduce(xm.REDUCE_SUM, x)
    
    start = (rank + OFF) % B
    
    # Create binary mask
    mask = torch.zeros_like(all_sum)
    if start <= B - 3:
        mask[start*S:(start+3)*S] = 1
    else:
        first_count = B - start
        mask[start*S:] = 1
        mask[:(3-first_count)*S] = 1
    
    buf = all_sum * mask
    
    return xm.all_reduce(xm.REDUCE_SUM, buf)
