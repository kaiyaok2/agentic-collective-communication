
def r67b_vself_b0p3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size; BETA = 0.3; N = 16384
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    # Try computing without explicit v vector
    s_even = s[0::2]
    s_odd = s[1::2]
    correction = (s_even.sum() - s_odd.sum()) / N
    # Reconstruct with alternating correction
    buf = s.clone()
    buf[0::2] = s_even + BETA * correction
    buf[1::2] = s_odd - BETA * correction
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    return acc / W
