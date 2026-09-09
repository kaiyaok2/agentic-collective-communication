"""
Profiling integration for AllToAllV schedule evaluation.

Generates per-step timing breakdowns from the simulator, identifying
bottleneck steps and link saturation. This profiling data is fed back
to the LLM in the CGIS refinement loop, giving it concrete timing
evidence alongside contention analysis.

Key insight: the simulator already produces per-step times, but they
were only used for aggregate scoring. By surfacing per-step data to
the LLM, it can see *which* steps dominate latency and make targeted
reordering decisions.
"""

from collections import defaultdict
from simulator.topology import TrainiumTopology
from simulator.alltoallv import AllToAllVSimulator


class ProfilingResult:
    """Per-step profiling data for a schedule evaluation."""

    def __init__(self, template, schedule, step_times, step_details,
                 total_time_us, lower_bound_us):
        self.template = template
        self.schedule = schedule
        self.step_times = step_times          # list of floats (seconds)
        self.step_details = step_details      # list of dicts with per-step info
        self.total_time_us = total_time_us
        self.lower_bound_us = lower_bound_us

    @property
    def num_steps(self):
        return len(self.step_times)

    def bottleneck_steps(self, top_k=5):
        """Return indices into step_details of the slowest steps."""
        indexed = sorted(enumerate(self.step_details),
                         key=lambda x: x[1].get("time_us", 0), reverse=True)
        return [idx for idx, _ in indexed[:top_k]]

    def step_time_us(self, idx):
        return self.step_details[idx].get("time_us", 0)

    def step_pct(self, idx):
        total = sum(d.get("time_us", 0) for d in self.step_details)
        if total == 0:
            return 0.0
        return (self.step_details[idx].get("time_us", 0) / total) * 100.0

    def efficiency(self):
        if self.total_time_us == 0:
            return 0.0
        return self.lower_bound_us / self.total_time_us


def profile_schedule(template, params, send_counts, topology,
                     element_bytes=4):
    """
    Profile a schedule and return detailed per-step timing breakdown.

    Returns a ProfilingResult with per-step timing data, link utilization
    details, and bottleneck identification.
    """
    sim = AllToAllVSimulator(topology, send_counts, element_bytes)
    lb = sim.lower_bound()

    total_time, step_times = sim.simulate_template(template, params)

    step_details = _analyze_steps(template, params, topology, send_counts,
                                  step_times, element_bytes)

    return ProfilingResult(
        template=template,
        schedule=_extract_schedule(template, params),
        step_times=step_times,
        step_details=step_details,
        total_time_us=total_time * 1e6,
        lower_bound_us=lb * 1e6,
    )


def _extract_schedule(template, params):
    if template == "permute_ring":
        return params["schedule"]
    elif template == "hierarchical":
        return params["inter_schedule"]
    elif template == "pairwise":
        return params.get("round_order", [])
    elif template == "hybrid_ag_perm":
        return params.get("permute_schedule", [])
    elif template == "multinode_hierarchical":
        return (params.get("intra_node_schedule", []),
                params.get("inter_node_schedule", []))
    elif template == "node_allgather":
        return []
    return []


def _analyze_steps(template, params, topology, send_counts,
                   step_times, element_bytes):
    """Generate per-step link utilization and bottleneck info."""
    world = topology.num_cores
    details = []

    if template == "permute_ring":
        schedule = params["schedule"]
        for step_idx, d in enumerate(schedule):
            link_usage = defaultdict(int)
            total_hops = 0
            for r in range(world):
                dst = (r + d) % world
                if r == dst:
                    continue
                src_dev = topology.rank_to_device(r)
                dst_dev = topology.rank_to_device(dst)
                if src_dev == dst_dev:
                    continue
                path = topology.device_path(src_dev, dst_dev)
                total_hops += len(path) - 1
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    link_usage[key] += 1

            max_contention = max(link_usage.values()) if link_usage else 0
            busiest_link = (max(link_usage, key=link_usage.get)
                           if link_usage else None)
            details.append({
                "step": step_idx,
                "distance": d,
                "time_us": step_times[step_idx] * 1e6,
                "max_link_contention": max_contention,
                "busiest_link": busiest_link,
                "total_hops": total_hops,
                "num_links_used": len(link_usage),
            })

    elif template == "hierarchical":
        inter_schedule = params["inter_schedule"]
        num_devices = topology.num_devices
        # step_times[0] is intra-device (free), steps 1+ are inter-device
        for step_idx, d in enumerate(inter_schedule):
            st_idx = step_idx + 1  # offset for intra-device step
            link_usage = defaultdict(int)
            total_hops = 0
            for dev in range(num_devices):
                dst_dev = (dev + d) % num_devices
                if dev == dst_dev:
                    continue
                path = topology.device_path(dev, dst_dev)
                total_hops += len(path) - 1
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                    link_usage[key] += 1

            max_contention = max(link_usage.values()) if link_usage else 0
            busiest_link = (max(link_usage, key=link_usage.get)
                           if link_usage else None)
            t_us = step_times[st_idx] * 1e6 if st_idx < len(step_times) else 0
            details.append({
                "step": step_idx,
                "distance": d,
                "time_us": t_us,
                "max_link_contention": max_contention,
                "busiest_link": busiest_link,
                "total_hops": total_hops,
                "num_links_used": len(link_usage),
            })
    elif template == "multinode_hierarchical":
        intra_schedule = params.get("intra_node_schedule", [])
        inter_schedule = params.get("inter_node_schedule", [])
        num_nodes = getattr(topology, 'num_nodes', 1)
        devices_per_node = getattr(topology, 'devices_per_node',
                                    topology.num_devices)

        # step_times[0] = intra-device (free)
        # steps 1..len(intra_schedule) = intra-node inter-device
        # remaining steps = inter-node
        st_offset = 1  # skip intra-device step

        for step_idx, d in enumerate(intra_schedule):
            st_idx = st_offset + step_idx
            link_usage = defaultdict(int)
            total_hops = 0
            for node_id in range(num_nodes):
                base_dev = node_id * devices_per_node
                for dev_off in range(devices_per_node):
                    src_dev = base_dev + dev_off
                    dst_dev = base_dev + (dev_off + d) % devices_per_node
                    if src_dev == dst_dev:
                        continue
                    path = topology.device_path(src_dev, dst_dev)
                    total_hops += len(path) - 1
                    for i in range(len(path) - 1):
                        key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
                        link_usage[key] += 1

            max_contention = max(link_usage.values()) if link_usage else 0
            busiest_link = (max(link_usage, key=link_usage.get)
                           if link_usage else None)
            t_us = step_times[st_idx] * 1e6 if st_idx < len(step_times) else 0
            details.append({
                "step": step_idx,
                "distance": d,
                "time_us": t_us,
                "max_link_contention": max_contention,
                "busiest_link": busiest_link,
                "total_hops": total_hops,
                "num_links_used": len(link_usage),
                "is_inter_node": False,
            })

        # Inter-node steps
        inter_offset = st_offset + len(intra_schedule)
        for step_idx, nd in enumerate(inter_schedule):
            st_idx = inter_offset + step_idx
            t_us = step_times[st_idx] * 1e6 if st_idx < len(step_times) else 0
            details.append({
                "step": len(intra_schedule) + step_idx,
                "distance": nd,
                "time_us": t_us,
                "is_inter_node": True,
                "node_distance": nd,
            })

    else:
        # For other templates, provide basic timing
        for step_idx, st in enumerate(step_times):
            details.append({
                "step": step_idx,
                "time_us": st * 1e6,
            })

    return details


def format_profiling_report(result, top_k=5):
    """
    Format profiling results into LLM-readable text.

    Returns a string suitable for inclusion in CGIS prompts.
    """
    lines = []
    lines.append(f"## Profiling Report")
    lines.append(f"Total latency: {result.total_time_us:.1f} us "
                 f"(lower bound: {result.lower_bound_us:.1f} us, "
                 f"efficiency: {result.efficiency():.1%})")
    lines.append(f"Steps: {result.num_steps}")
    lines.append("")

    # Bottleneck steps
    bottlenecks = result.bottleneck_steps(top_k)
    lines.append(f"### Slowest {min(top_k, len(bottlenecks))} steps "
                 f"(dominate total latency):")
    lines.append("")
    for rank_pos, idx in enumerate(bottlenecks):
        d = result.step_details[idx]
        pct = result.step_pct(idx)
        line = (f"  {rank_pos+1}. Step {d['step']}")
        if "distance" in d:
            line += f" (distance={d['distance']})"
        line += f": {d['time_us']:.1f} us ({pct:.1f}% of total)"
        if "max_link_contention" in d:
            line += f", link_contention={d['max_link_contention']}"
        if "busiest_link" in d:
            line += f", busiest_link={d['busiest_link']}"
        if "total_hops" in d:
            line += f", hops={d['total_hops']}"
        lines.append(line)

    lines.append("")

    # Full step timing table
    lines.append("### All steps (sorted by schedule order):")
    lines.append("")
    for d in result.step_details:
        pct = result.step_pct(d["step"])
        is_bottleneck = d["step"] in bottlenecks
        marker = " <<<" if is_bottleneck else ""
        line = f"  Step {d['step']:2d}"
        if "distance" in d:
            line += f" dist={d['distance']:2d}"
        line += f": {d['time_us']:7.1f} us ({pct:4.1f}%)"
        if d.get("is_inter_node"):
            line += "  [INTER-NODE]"
        if "max_link_contention" in d:
            line += f"  contention={d['max_link_contention']}"
        line += marker
        lines.append(line)

    lines.append("")

    # Timing distribution summary
    times_us = [d["time_us"] for d in result.step_details]
    if times_us:
        lines.append("### Timing distribution:")
        mean_t = sum(times_us) / len(times_us)
        max_t = max(times_us)
        min_t = min(times_us)
        lines.append(f"  Mean: {mean_t:.1f} us, "
                     f"Min: {min_t:.1f} us, "
                     f"Max: {max_t:.1f} us, "
                     f"Spread: {max_t - min_t:.1f} us")
        # What fraction of time is in top 3 steps?
        sorted_times = sorted(times_us, reverse=True)
        top3_pct = sum(sorted_times[:3]) / sum(times_us) * 100 if sum(times_us) > 0 else 0
        lines.append(f"  Top 3 steps account for {top3_pct:.1f}% of total latency")

    return "\n".join(lines)
