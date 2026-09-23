
def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 1.0
    
    # All_reduce sums x across all ranks - all ranks get same result
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Since all ranks have same s, no need for second all_reduce
    # The second all_reduce(sum) / W just returns buf since all ranks have identical buf
    acc = s + BETA * s.mean()
    
    return acc
