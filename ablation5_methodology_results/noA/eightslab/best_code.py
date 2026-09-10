
def eightslab_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    n_slabs, slab_N = 8, 1024
    # Single all_reduce on entire tensor (instead of 8 separate ones)
    reduced = xm.all_reduce(xm.REDUCE_SUM, x)
    # Create scale factors: [1,1,...,1, 2,2,...,2, ..., 8,8,...,8]
    indices = torch.arange(n_slabs * slab_N, device=x.device, dtype=x.dtype)
    scale_factors = indices // slab_N + 1
    return reduced * scale_factors
