
def evolved_p5601(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try to reduce ops by returning without final view
    if N * world_size == x.shape[0]:
        # Extract just my chunk to send
        my_chunk = x[rank * N:(rank + 1) * N]
        
        # Gather this chunk from all ranks
        gathered = xm.all_gather(my_chunk.unsqueeze(0), dim=0)
        return gathered.reshape(-1)
    else:
        gathered = xm.all_gather(x.unsqueeze(0), dim=0)
        return gathered[rank]
