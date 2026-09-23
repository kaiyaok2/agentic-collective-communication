
def r60c_b9_L3_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; B = 9; OFF = 2
    
    # Try creating mask using arange and comparison
    start = (rank + OFF) % B
    buckets = torch.arange(B * S, device=x.device, dtype=torch.long) // S
    keep_mask = ((buckets == start % B) | (buckets == (start+1) % B) | (buckets == (start+2) % B))
    mask = keep_mask.to(x.dtype)
    
    return xm.all_reduce(xm.REDUCE_SUM, xm.all_reduce(xm.REDUCE_SUM, x) * mask)
