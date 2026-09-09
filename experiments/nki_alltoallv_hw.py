"""
NKI AllToAllV kernel for real Trainium hardware benchmarking.

Generates @nki.jit kernels with fully unrolled pack+extract loops.
The entire AllToAllV (pack -> all_gather -> extract) runs in a single
NKI kernel on the NeuronDevice — zero XLA ops.
"""
import importlib
import os
import sys
import tempfile


def make_nki_alltoallv_kernel(send_counts_matrix, world_size, rank):
    """Create an @nki.jit AllToAllV kernel that does pack+gather+extract.

    Returns (kernel_fn, recv_total) where kernel_fn is an @nki.jit function
    that takes (1, total_send) input and returns (1, recv_total) output.
    All operations happen inside the NKI kernel.
    """
    send_counts = send_counts_matrix[rank]
    recv_counts = [send_counts_matrix[src][rank] for src in range(world_size)]
    max_chunk = max(
        max(send_counts_matrix[s][d]
            for s in range(world_size) for d in range(world_size)),
        1,
    )

    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)

    recv_total = sum(recv_counts)
    pack_size = world_size * max_chunk

    # Pack operations: (src_offset, dst_offset, count)
    pack_ops = []
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            pack_ops.append((send_offsets[i], i * max_chunk, sc))

    # Extract operations: (gathered_offset, output_offset, count)
    extract_ops = []
    out_offset = 0
    for src in range(world_size):
        count = recv_counts[src]
        if count > 0:
            g_offset = src * pack_size + rank * max_chunk
            extract_ops.append((g_offset, out_offset, count))
        out_offset += count

    # Generate kernel source with fully unrolled operations
    lines = [
        'import numpy as np',
        'import neuronxcc.nki as nki',
        'import neuronxcc.nki.language as nl',
        'import neuronxcc.nki.nccl as nccl',
        '',
        '@nki.jit',
        'def nki_alltoallv(input_hbm):',
        f'    replica_groups = [list(range({world_size}))]',
        '',
        '    # Pack: copy data for each destination into canonical slots',
        f'    packed = nl.ndarray((1, {pack_size}), dtype=nl.float32, buffer=nl.shared_hbm)',
    ]

    for so, do, sc in pack_ops:
        lines.append(f'    nl.store(packed[0:1, {do}:{do + sc}], nl.load(input_hbm[0:1, {so}:{so + sc}]))')

    lines.extend([
        '',
        '    # All_gather: replicate all packed buffers',
        f'    gathered = nl.ndarray((1, {world_size * pack_size}), dtype=nl.float32, buffer=nl.shared_hbm)',
        '    nccl.all_gather(op=np.add, srcs=[packed], dsts=[gathered],',
        '                    replica_groups=replica_groups,',
        '                    all_gather_dim=1, dtype=nl.float32)',
        '',
        '    # Extract: pull out data destined for this rank',
        f'    output = nl.ndarray((1, {recv_total}), dtype=nl.float32, buffer=nl.shared_hbm)',
    ])

    for go, oo, count in extract_ops:
        lines.append(f'    nl.store(output[0:1, {oo}:{oo + count}], nl.load(gathered[0:1, {go}:{go + count}]))')

    lines.extend([
        '',
        '    return output',
    ])

    src = '\n'.join(lines) + '\n'

    # Write to temp file so inspect.getsource works for @nki.jit
    tmpdir = tempfile.mkdtemp(prefix='nki_kernel_')
    tmpfile = os.path.join(tmpdir, f'nki_kernel_r{rank}.py')
    with open(tmpfile, 'w') as f:
        f.write(src)

    mod_name = f'nki_kernel_r{rank}'
    spec = importlib.util.spec_from_file_location(mod_name, tmpfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    return mod.nki_alltoallv, recv_total, pack_size, None


def make_nki_alltoallv_kernel_gather_only(send_counts_matrix, world_size, rank):
    """NKI kernel that only does pack+gather. Extraction done with torch.

    This variant is useful for comparing NKI gather vs XLA gather overhead.
    Returns (kernel_fn, recv_total, pack_size, flat_extract_indices).
    """
    send_counts = send_counts_matrix[rank]
    recv_counts = [send_counts_matrix[src][rank] for src in range(world_size)]
    max_chunk = max(
        max(send_counts_matrix[s][d]
            for s in range(world_size) for d in range(world_size)),
        1,
    )

    send_offsets = [0]
    for c in send_counts[:-1]:
        send_offsets.append(send_offsets[-1] + c)

    recv_total = sum(recv_counts)
    pack_size = world_size * max_chunk

    flat_idx = []
    for src in range(world_size):
        count = recv_counts[src]
        base = src * pack_size + rank * max_chunk
        flat_idx.extend(range(base, base + count))

    pack_ops = []
    for i in range(world_size):
        sc = send_counts[i]
        if sc > 0:
            pack_ops.append((send_offsets[i], i * max_chunk, sc))

    lines = [
        'import numpy as np',
        'import neuronxcc.nki as nki',
        'import neuronxcc.nki.language as nl',
        'import neuronxcc.nki.nccl as nccl',
        '',
        '@nki.jit',
        'def nki_alltoallv_gather(input_hbm):',
        f'    replica_groups = [list(range({world_size}))]',
        f'    packed = nl.ndarray((1, {pack_size}), dtype=nl.float32, buffer=nl.shared_hbm)',
    ]

    for so, do, sc in pack_ops:
        lines.append(f'    nl.store(packed[0:1, {do}:{do + sc}], nl.load(input_hbm[0:1, {so}:{so + sc}]))')

    lines.extend([
        f'    gathered = nl.ndarray((1, {world_size * pack_size}), dtype=nl.float32, buffer=nl.shared_hbm)',
        '    nccl.all_gather(op=np.add, srcs=[packed], dsts=[gathered],',
        '                    replica_groups=replica_groups,',
        '                    all_gather_dim=1, dtype=nl.float32)',
        '    return gathered',
    ])

    src = '\n'.join(lines) + '\n'
    tmpdir = tempfile.mkdtemp(prefix='nki_kernel_')
    tmpfile = os.path.join(tmpdir, f'nki_kernel_go_r{rank}.py')
    with open(tmpfile, 'w') as f:
        f.write(src)

    mod_name = f'nki_kernel_go_r{rank}'
    spec = importlib.util.spec_from_file_location(mod_name, tmpfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    return mod.nki_alltoallv_gather, recv_total, pack_size, flat_idx
