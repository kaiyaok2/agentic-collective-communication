
def r61c_dd_meansq3_p512_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 512
    B = 8
    
    # All_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x).view(B, S)
    
    # Since s is same on all ranks, f is same, so s*f is same
    # all_reduce(s*f) would give world_size * (s*f)
    # Dividing by world_size gives back s*f
    f = 1.0 + 3.0 * (s * s).mean(dim=1, keepdim=True)
    result = s * f
    
    return result.view(-1)
