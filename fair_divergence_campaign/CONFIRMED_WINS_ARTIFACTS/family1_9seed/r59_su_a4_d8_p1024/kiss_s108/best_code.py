
def r59_su_a4_d8_p1024_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 1024; W = world_size
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Try building weights using torch.ones
    weight_tensor = torch.ones(W * S, device=x.device, dtype=x.dtype)
    for r in range(W):
        weight_tensor[r*S:(r+1)*S] = a[r] / W
    buf = s * weight_tensor
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
