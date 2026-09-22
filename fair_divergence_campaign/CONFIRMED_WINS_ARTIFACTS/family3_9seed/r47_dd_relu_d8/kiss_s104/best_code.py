
def r47_dd_relu_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    s_view = s.view(8, 256)
    means = s_view.mean(dim=1, keepdim=True)
    buf = (s_view * (1.0 + means * (means > 0).to(s.dtype))).view(-1)
    
    return buf
