
def evolved_p9001(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: a = zeros; for i in 0..7:
    #   a += MAXreduce(x) * ((i+1)*0.1)
    #   a += MINreduce(x) * ((i+1)*0.05)
    # 
    # Optimization: hoist invariant reductions out of loop
    # Sum of coefficients for MAX: 0.1*(1+2+3+4+5+6+7+8) = 3.6
    # Sum of coefficients for MIN: 0.05*(1+2+3+4+5+6+7+8) = 1.8
    
    max_x = xm.all_reduce(xm.REDUCE_MAX, x)
    min_x = xm.all_reduce(xm.REDUCE_MIN, x)
    a = max_x * 3.6 + min_x * 1.8
    return a
