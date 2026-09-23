
def r66b_walsh_b0p7_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    idx = torch.arange(N, device=x.device, dtype=x.dtype)
    v = 1.0 - 2.0 * (idx % 2)
    u_beta = 0.7 - 1.4 * ((idx // 2) % 2)
    inv_W = 1.0 / W
    
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    s = acc - u_beta * (v * acc).mean()
    acc = xm.all_reduce(xm.REDUCE_SUM, s + u_beta * (v * s).mean()) * inv_W
    
    return acc
