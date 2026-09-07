
def evolved_p4401(x1, x2, x3, x4, x5, x6, x7, x8, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single packed all-reduce
    packed = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8], dim=0)
    reduced = xm.all_reduce(xm.REDUCE_SUM, packed)
    
    r1 = reduced[0:N]
    r2 = reduced[N:2*N]
    r3 = reduced[2*N:3*N]
    r4 = reduced[3*N:4*N]
    r5 = reduced[4*N:5*N]
    r6 = reduced[5*N:6*N]
    r7 = reduced[6*N:7*N]
    r8 = reduced[7*N:8*N]
    
    s = 0.5 * r1 + 1.5 * r2 + 2.5 * r3 + 3.5 * r4 + 4.5 * r5 + 5.5 * r6 + 6.5 * r7 + 7.5 * r8
    return s
