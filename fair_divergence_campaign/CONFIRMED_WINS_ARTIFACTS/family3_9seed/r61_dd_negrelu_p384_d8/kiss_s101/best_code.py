
def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # After first all_reduce, all ranks have the same data
    # So the second all_reduce just multiplies by world_size
    # Then we divide by world_size, which cancels out!
    import torch.nn.functional as F
    factors = 1.0 + F.relu(-s.view(B, S).mean(dim=1))
    result = s * factors.repeat_interleave(S)
    
    return result
