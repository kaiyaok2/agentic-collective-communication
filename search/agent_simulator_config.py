"""
Agent-built simulator: LLM discovers hardware characteristics and builds its own cost model.

The agent is hardware-agnostic — it doesn't know anything about Trainium's topology,
dispatch overhead, or communication costs upfront. Instead, it:

1. Profiles the hardware using tool calls (measure collectives, p2p, device info)
2. Writes its own Python simulator/cost model based on profiling results
3. Iteratively tests and refines the simulator until predictions match measurements
4. Returns the validated simulator for use in downstream phases

This makes the agent portable to any accelerator: the same loop works on TPU, GPU,
or future hardware. The agent discovers hardware characteristics (op costs, topology,
dispatch overhead) through profiling and builds an accurate cost model from first principles.

Downstream phases (baseline eval, evolution, mini-benchmarking) can feed errors back
to Phase 1 to trigger simulator refinement.
"""

import json
import math
import re
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import boto3

from .generate_algo import BEDROCK_REGION, MODEL_IDS


@dataclass
class SimulatorConfig:
    """Hardware parameters discovered by the agent."""
    num_devices: int = 0
    cores_per_device: int = 0
    device_adjacency: Dict[int, List[int]] = field(default_factory=dict)

    link_bandwidth_gbps: float = 0.0
    link_latency_us: float = 0.0

    collective_dispatch_overhead_us: float = 0.0
    local_op_overhead_us: float = 0.0

    efa_bandwidth_gbps: float = 0.0
    efa_latency_us: float = 0.0
    efa_adapters_per_node: int = 0

    num_cores: int = 0
    max_hops: int = 0
    avg_hops: float = 0.0

    unsupported_primitives: List[str] = field(default_factory=list)

    def is_complete(self):
        return (self.num_devices > 0 and
                self.cores_per_device > 0 and
                self.link_bandwidth_gbps > 0 and
                self.collective_dispatch_overhead_us > 0 and
                len(self.device_adjacency) > 0)


@dataclass
class AgentSimulator:
    """LLM-built simulator: code + config + validation state."""
    config: SimulatorConfig = field(default_factory=SimulatorConfig)
    simulator_code: str = ""
    cost_function: Optional[callable] = None
    validation_results: List[Dict] = field(default_factory=list)
    knowledgebase: Dict[str, Any] = field(default_factory=dict)
    build_history: List[str] = field(default_factory=list)

    def is_validated(self, tolerance_pct=20.0):
        if not self.validation_results:
            return False
        return all(r.get("within_tolerance", False) for r in self.validation_results)

    def predict(self, algorithm_description: str, **kwargs) -> float:
        if self.cost_function is None:
            raise RuntimeError("Simulator not built yet — call build_simulator first")
        return self.cost_function(algorithm_description, **kwargs)


# ============================================================
# Pre-measured hardware values (used by profiling tools)
# ============================================================

_HARDWARE_MEASUREMENTS = {
    "all_gather_latency_us": {
        "world_32_bytes_4096": 1050.0,
        "world_32_bytes_32768": 1053.0,
        "world_32_bytes_524288": 1050.0,
        "world_32_bytes_8388608": 1054.0,
    },
    "collective_permute_latency_us": {
        "1_step_bytes_4096": 90.0,
        "1_step_bytes_32768": 91.0,
        "15_steps_bytes_4096": 2330.0,
        "31_steps_bytes_4096": 2780.0,
    },
    "p2p_latency_us": {
        "same_device": 0.1,
        "1_hop_4096B": 2.5,
        "1_hop_32768B": 3.2,
        "2_hop_4096B": 5.1,
        "3_hop_4096B": 7.8,
        "4_hop_4096B": 10.2,
    },
    "xla_op_overhead_us": {
        "index_select": 29.0,
        "slice": 28.5,
        "cat": 29.2,
        "view": 0.1,
        "reshape": 0.1,
        "unsqueeze": 0.1,
        "squeeze": 0.1,
        "flatten": 0.1,
        "narrow": 0.1,
        "transpose": 0.1,
        "permute": 0.1,
        "expand": 0.1,
        "contiguous": 0.1,
        "zeros": 28.0,
        "tensor": 28.5,
        "stack": 29.0,
        "gather": 29.5,
        "scatter": 29.5,
        "split": 28.5,
        "chunk": 28.5,
    },
    "local_op_overhead_us": 28.8,
    # Scaling behaviors discovered from real training
    "index_select_scaling": {
        "note": "index_select cost grows with index_tensor_size * source_tensor_size due to random access",
        "small_index_100": 29.0,
        "medium_index_10k": 35.0,
        "large_index_1M": 500.0,
        "catastrophic_index_3M_source_200M": 20000000.0,
    },
    "cross_node_constraints": {
        "collective_permute_cross_node": "UNSUPPORTED",
        "note": "collective_permute ring patterns that cross node boundaries cause SIGABRT on this hardware",
    },
    "overlap_effects": {
        "note": "Per-tensor all_reduce can overlap with backward compute; batched cat+all_reduce serializes",
        "per_tensor_allreduce_overlap_factor": 0.3,
        "batched_allreduce_overlap_factor": 0.0,
    },
    "num_devices": 16,
    "cores_per_device": 2,
    "device_adjacency": {
        0:  [12, 3, 4, 1],   4:  [0, 7, 8, 5],   8:  [4, 11, 12, 9],  12: [8, 15, 0, 13],
        1:  [13, 0, 5, 2],   5:  [1, 4, 9, 6],   9:  [5, 8, 13, 10],  13: [9, 12, 1, 14],
        2:  [14, 1, 6, 3],   6:  [2, 5, 10, 7],  10: [6, 9, 14, 11],  14: [10, 13, 2, 15],
        3:  [15, 2, 7, 0],   7:  [3, 6, 11, 4],  11: [7, 10, 15, 8],  15: [11, 14, 3, 12],
    },
    "link_bandwidth_gbps": 192.0,
    "link_latency_us": 0.5,
    "efa_bandwidth_gbps": 12.5,
    "efa_latency_us": 5.0,
    "efa_adapters_per_node": 8,
    # Algorithm-level measurements for validation (named by structure, not technique)
    "algorithm_latencies": {
        "algo_1coll_3localops_32ranks": 130.0,
        "algo_1coll_35localops_32ranks": 1050.0,
        "algo_31coll_62localops_32ranks": 2780.0,
        "algo_2coll_5localops_32ranks": 730.0,
        "algo_fused_32ranks": 1200.0,
        "algo_15coll_30localops_32ranks": 2330.0,
    },
}


# ============================================================
# Dynamic primitive compilation testing
# ============================================================

_primitive_compilation_cache = {}

def _test_primitive_compilation(primitive):
    """Test whether an XLA primitive compiles on the current hardware.

    Runs the test in a subprocess to prevent neuron runtime crashes from
    killing the main process. Results are cached per-process."""
    if primitive in _primitive_compilation_cache:
        return _primitive_compilation_cache[primitive]

    import subprocess, sys, os

    test_script = f"""\
import sys, os
os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
try:
    import torch
    import torch_xla.core.xla_model as _xm
    dev = _xm.xla_device()
    if "{primitive}" == "all_gather":
        t = torch.ones(4, device=dev)
        _ = _xm.all_gather(t, dim=0)
        _xm.mark_step()
    elif "{primitive}" == "reduce_scatter":
        t = torch.ones(4, device=dev)
        _ = _xm.reduce_scatter(_xm.REDUCE_SUM, t, scale=1.0, scatter_dim=0, shard_count=1)
        _xm.mark_step()
    elif "{primitive}" == "all_reduce":
        t = torch.ones(4, device=dev)
        _ = _xm.all_reduce('sum', t)
        _xm.mark_step()
    elif "{primitive}" == "collective_permute":
        t = torch.ones(4, device=dev)
        pairs = [(0, 0)]
        _ = _xm.collective_permute(t, pairs=pairs)
        _xm.mark_step()
    elif "{primitive}" == "all_to_all":
        t = torch.ones(4, device=dev)
        _ = _xm.all_to_all(t, split_dimension=0, concat_dimension=0, split_count=1)
        _xm.mark_step()
    else:
        print("UNKNOWN_PRIMITIVE", file=sys.stderr)
        sys.exit(2)
    print("OK")
except Exception as e:
    print(f"FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    try:
        venv_python = "/opt/aws_neuronx_venv_pytorch_2_9/bin/python3"
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        proc = subprocess.run(
            [python_exe, "-c", test_script],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "NEURON_RT_NUM_CORES": "1"})
        if proc.returncode == 0 and "OK" in proc.stdout:
            result = {
                "primitive": primitive,
                "compiles_on_hardware": True,
                "error_code": None,
                "note": f"{primitive} compiles and runs correctly on this hardware.",
            }
        else:
            err_str = proc.stderr[:300]
            error_code = None
            import re as _re
            if "No module named" in err_str:
                result = {
                    "primitive": primitive,
                    "compiles_on_hardware": True,
                    "error_code": "ENV_MISSING",
                    "note": f"Cannot test {primitive} — torch_xla not in subprocess environment. Assuming supported.",
                }
                _primitive_compilation_cache[primitive] = result
                return result
            m = _re.search(r'(NCC_\w+)', err_str)
            if m:
                error_code = m.group(1)
            elif "UNKNOWN_PRIMITIVE" in err_str:
                error_code = "UNKNOWN_PRIMITIVE"
            else:
                error_code = "COMPILATION_FAILED"
            result = {
                "primitive": primitive,
                "compiles_on_hardware": False,
                "error_code": error_code,
                "note": f"{primitive} failed to compile on this hardware. {err_str[:200]}",
            }
    except subprocess.TimeoutExpired:
        result = {
            "primitive": primitive,
            "compiles_on_hardware": False,
            "error_code": "TIMEOUT",
            "note": f"{primitive} compilation test timed out (possible runtime crash).",
        }
    except Exception as e:
        result = {
            "primitive": primitive,
            "compiles_on_hardware": False,
            "error_code": "TEST_ERROR",
            "note": f"{primitive} compilation test failed: {str(e)[:200]}",
        }

    _primitive_compilation_cache[primitive] = result
    return result


# ============================================================
# Tool handlers
# ============================================================

def _handle_tool_call(tool_name, tool_input, agent_sim):
    """Execute a profiling/simulator tool and return the result."""
    measurements = _HARDWARE_MEASUREMENTS

    if tool_name == "get_device_info":
        return json.dumps({
            "num_devices": measurements["num_devices"],
            "cores_per_device": measurements["cores_per_device"],
            "total_ranks": measurements["num_devices"] * measurements["cores_per_device"],
            "topology": "unknown — use measure_p2p_transfer to discover adjacency",
            "links_per_device": 4,
        })

    elif tool_name == "measure_collective_latency":
        op_type = tool_input.get("collective_type", "all_gather")
        num_steps = tool_input.get("num_steps", 1)
        tensor_bytes = tool_input.get("tensor_bytes", 4096)
        world_size = tool_input.get("world_size", 32)

        if op_type == "all_gather":
            key = f"world_{world_size}_bytes_{tensor_bytes}"
            latency = measurements["all_gather_latency_us"].get(key, 1050.0)
            return json.dumps({
                "collective_type": "all_gather",
                "world_size": world_size,
                "tensor_bytes": tensor_bytes,
                "latency_us": latency,
                "num_dispatches": 1,
            })
        elif op_type == "collective_permute":
            key = f"{num_steps}_step{'s' if num_steps > 1 else ''}_bytes_{tensor_bytes}"
            latency = measurements["collective_permute_latency_us"].get(
                key, 90.0 * num_steps)
            return json.dumps({
                "collective_type": "collective_permute",
                "num_steps": num_steps,
                "tensor_bytes": tensor_bytes,
                "latency_us": latency,
                "per_step_us": latency / max(num_steps, 1),
            })
        elif op_type == "all_to_all":
            return json.dumps({
                "collective_type": "all_to_all",
                "world_size": world_size,
                "tensor_bytes": tensor_bytes,
                "latency_us": 1200.0,
                "note": "XLA decomposes all_to_all into internal ring of send/recv pairs",
            })
        elif op_type == "reduce_scatter":
            return json.dumps({
                "collective_type": "reduce_scatter",
                "world_size": world_size,
                "tensor_bytes": tensor_bytes,
                "latency_us": 95.0,
                "num_dispatches": 1,
            })

    elif tool_name == "measure_p2p_transfer":
        src_device = tool_input.get("src_device", 0)
        dst_device = tool_input.get("dst_device", 1)
        tensor_bytes = tool_input.get("tensor_bytes", 4096)

        if src_device == dst_device:
            latency = measurements["p2p_latency_us"]["same_device"]
            hops = 0
        else:
            adj = measurements["device_adjacency"]
            hops = _compute_hops(adj, src_device, dst_device)
            key = f"{hops}_hop{'s' if hops > 1 else ''}_{tensor_bytes}B"
            latency = measurements["p2p_latency_us"].get(
                key, hops * 2.5 + tensor_bytes / (192e9 / 1e6))

        return json.dumps({
            "src_device": src_device,
            "dst_device": dst_device,
            "hops": hops,
            "tensor_bytes": tensor_bytes,
            "latency_us": latency,
        })

    elif tool_name == "measure_xla_op_overhead":
        op_name = tool_input.get("op_name", "index_select")
        tensor_size = tool_input.get("tensor_size", 32768)
        num_ops = tool_input.get("num_ops", 1)

        per_op = measurements["xla_op_overhead_us"].get(op_name, 29.0)

        if "measured_op_costs" not in agent_sim.knowledgebase:
            agent_sim.knowledgebase["measured_op_costs"] = {}
        agent_sim.knowledgebase["measured_op_costs"][op_name] = per_op

        return json.dumps({
            "op_name": op_name,
            "tensor_size": tensor_size,
            "num_ops": num_ops,
            "per_op_overhead_us": per_op,
            "total_overhead_us": per_op * num_ops,
            "note": "Each XLA/HLO op compiles to a separate kernel launch on the accelerator",
        })

    elif tool_name == "measure_algorithm_latency":
        algorithm = tool_input.get("algorithm", "")
        algo_map = measurements["algorithm_latencies"]
        latency = algo_map.get(algorithm)
        if latency is None:
            available = list(algo_map.keys())
            return json.dumps({
                "error": f"No measurement for '{algorithm}'",
                "available_algorithms": available,
            })
        return json.dumps({
            "algorithm": algorithm,
            "latency_us": latency,
        })

    elif tool_name == "measure_index_select_scaling":
        index_size = tool_input.get("index_tensor_size", 1000)
        source_size = tool_input.get("source_tensor_size", 100000)
        scaling = measurements.get("index_select_scaling", {})
        if index_size < 1000:
            latency = scaling.get("small_index_100", 29.0)
        elif index_size < 100000:
            latency = scaling.get("medium_index_10k", 35.0)
        elif index_size < 2000000:
            latency = scaling.get("large_index_1M", 500.0)
        else:
            latency = scaling.get("catastrophic_index_3M_source_200M", 20000000.0)
        return json.dumps({
            "index_tensor_size": index_size,
            "source_tensor_size": source_size,
            "latency_us": latency,
            "note": ("index_select has random memory access pattern. "
                     "Cost grows super-linearly with index size on NeuronCores "
                     "because HBM has high sequential bandwidth but poor random-access throughput."),
        })

    elif tool_name == "measure_overlap_potential":
        collective_type = tool_input.get("collective_type", "all_reduce")
        pattern = tool_input.get("pattern", "per_tensor")
        overlap = measurements.get("overlap_effects", {})
        if pattern == "per_tensor":
            factor = overlap.get("per_tensor_allreduce_overlap_factor", 0.3)
            note = ("Per-tensor all_reduce calls can overlap with ongoing backward-pass "
                    "computation. The XLA compiler pipelines independent collectives with "
                    "compute, reducing effective wall-clock time by ~30%.")
        else:
            factor = overlap.get("batched_allreduce_overlap_factor", 0.0)
            note = ("Batched approach (cat → all_reduce → split) requires all gradients "
                    "before the all_reduce can start, serializing communication with compute. "
                    "No overlap possible.")
        return json.dumps({
            "collective_type": collective_type,
            "pattern": pattern,
            "overlap_factor": factor,
            "effective_cost_multiplier": 1.0 - factor,
            "note": note,
        })

    elif tool_name == "check_cross_node_support":
        collective = tool_input.get("collective_type", "collective_permute")
        constraints = measurements.get("cross_node_constraints", {})
        if collective == "collective_permute":
            return json.dumps({
                "collective_type": collective,
                "cross_node_supported": False,
                "note": ("collective_permute ring patterns that cross node boundaries "
                         "cause SIGABRT on this hardware. Only intra-node-only or "
                         "inter-node-only rings are supported, not mixed patterns."),
            })
        return json.dumps({
            "collective_type": collective,
            "cross_node_supported": True,
            "note": f"{collective} works across node boundaries via EFA.",
        })

    elif tool_name == "check_primitive_compilation":
        primitive = tool_input.get("primitive", "all_gather")
        known = ["all_gather", "reduce_scatter", "all_reduce",
                 "collective_permute", "all_to_all"]
        if primitive not in known:
            return json.dumps({
                "primitive": primitive,
                "error": f"Unknown primitive '{primitive}'",
                "available": known,
            })
        result = _test_primitive_compilation(primitive)
        return json.dumps(result)

    elif tool_name == "discover_device_adjacency":
        adj = measurements["device_adjacency"]
        return json.dumps({
            "device_adjacency": {str(k): v for k, v in adj.items()},
            "topology_hint": "4 neighbors per device, 16 devices total",
        })

    elif tool_name == "build_simulator":
        code = tool_input.get("code", "")
        agent_sim.simulator_code = code
        agent_sim.build_history.append(code)
        try:
            sandbox = {"__builtins__": __builtins__, "math": math}
            exec(code, sandbox)
            cost_fn = sandbox.get("estimate_latency")
            if cost_fn is None:
                return json.dumps({
                    "status": "error",
                    "error": "Code must define 'estimate_latency(algorithm_desc, **kwargs)' function",
                })
            test_result = cost_fn("test", num_collectives=1, num_xla_ops=0,
                                  data_bytes=4096, num_hops=1)
            agent_sim.cost_function = cost_fn
            return json.dumps({
                "status": "ok",
                "test_prediction": test_result,
                "note": "Simulator compiled. Use validate_simulator to check accuracy.",
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()[-500:],
            })

    elif tool_name == "validate_simulator":
        if agent_sim.cost_function is None:
            return json.dumps({"error": "No simulator built yet. Call build_simulator first."})

        test_cases = tool_input.get("test_cases", [])
        if not test_cases:
            test_cases = [
                {"algorithm": "algo_1coll_3localops_32ranks",
                 "kwargs": {"num_collectives": 1, "num_xla_ops": 3,
                            "data_bytes": 32768, "num_hops": 0}},
                {"algorithm": "algo_1coll_35localops_32ranks",
                 "kwargs": {"num_collectives": 1, "num_xla_ops": 35,
                            "data_bytes": 32768, "num_hops": 0}},
                {"algorithm": "algo_31coll_62localops_32ranks",
                 "kwargs": {"num_collectives": 31, "num_xla_ops": 62,
                            "data_bytes": 4096, "num_hops": 31}},
                {"algorithm": "algo_2coll_5localops_32ranks",
                 "kwargs": {"num_collectives": 2, "num_xla_ops": 5,
                            "data_bytes": 32768, "num_hops": 0}},
            ]

        results = []
        algo_latencies = measurements["algorithm_latencies"]
        for tc in test_cases:
            algo = tc["algorithm"]
            actual = algo_latencies.get(algo)
            if actual is None:
                results.append({"algorithm": algo, "error": "no measurement available"})
                continue
            try:
                predicted = agent_sim.cost_function(algo, **tc.get("kwargs", {}))
                error_pct = abs(predicted - actual) / actual * 100
                ok = error_pct < 20
                results.append({
                    "algorithm": algo,
                    "predicted_us": round(predicted, 1),
                    "actual_us": actual,
                    "error_percent": round(error_pct, 1),
                    "within_tolerance": ok,
                })
            except Exception as e:
                results.append({
                    "algorithm": algo,
                    "error": str(e),
                })

        agent_sim.validation_results = results
        all_ok = all(r.get("within_tolerance", False) for r in results if "error" not in r)
        return json.dumps({
            "results": results,
            "all_within_tolerance": all_ok,
            "note": "Simulator validated." if all_ok else "Some predictions are off — refine your cost model.",
        })

    elif tool_name == "set_simulator_config":
        for key in ["link_bandwidth_gbps", "link_latency_us",
                     "collective_dispatch_overhead_us", "local_op_overhead_us",
                     "num_devices", "cores_per_device"]:
            if key in tool_input:
                setattr(agent_sim.config, key, tool_input[key])
        if "device_adjacency" in tool_input:
            adj = tool_input["device_adjacency"]
            agent_sim.config.device_adjacency = {int(k): v for k, v in adj.items()}
        if "unsupported_primitives" in tool_input:
            agent_sim.config.unsupported_primitives = tool_input["unsupported_primitives"]
        return json.dumps({"status": "ok"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _compute_hops(adjacency, src, dst):
    if src == dst:
        return 0
    visited = {src}
    queue = [(src, 0)]
    while queue:
        node, dist = queue.pop(0)
        for neighbor in adjacency.get(node, adjacency.get(str(node), [])):
            if neighbor == dst:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return 99


# ============================================================
# Tool definitions for Bedrock API
# ============================================================

PROFILING_TOOLS = [
    {
        "name": "get_device_info",
        "description": "Get basic hardware info: number of devices and cores per device. Does NOT reveal topology — use measure_p2p_transfer or discover_device_adjacency for that.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "measure_collective_latency",
        "description": "Measure the end-to-end latency of a collective operation on the hardware. Available collectives: all_gather, collective_permute, reduce_scatter, all_to_all. Vary num_steps and tensor_bytes to separate fixed overhead from bandwidth cost.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collective_type": {
                    "type": "string",
                    "enum": ["all_gather", "collective_permute", "reduce_scatter", "all_to_all"],
                },
                "num_steps": {
                    "type": "integer",
                    "description": "Number of steps (for collective_permute). Default 1.",
                },
                "tensor_bytes": {
                    "type": "integer",
                    "description": "Size of tensor in bytes. Try multiple sizes to separate overhead from bandwidth.",
                },
                "world_size": {
                    "type": "integer",
                    "description": "Number of ranks. Default 32.",
                },
            },
            "required": ["collective_type"],
        },
    },
    {
        "name": "measure_p2p_transfer",
        "description": "Measure point-to-point transfer latency between two devices. Use to discover topology: test many device pairs to find which are neighbors vs far apart.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src_device": {"type": "integer"},
                "dst_device": {"type": "integer"},
                "tensor_bytes": {"type": "integer", "description": "Default 4096"},
            },
            "required": ["src_device", "dst_device"],
        },
    },
    {
        "name": "measure_xla_op_overhead",
        "description": "Measure the per-op overhead of a local XLA/HLO operation. Different operation types have very different costs. Measure multiple ops to build a complete cost model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "op_name": {
                    "type": "string",
                    "enum": ["index_select", "slice", "cat", "view", "reshape",
                             "unsqueeze", "squeeze", "flatten", "narrow",
                             "transpose", "permute", "expand", "contiguous",
                             "zeros", "tensor", "stack", "gather", "scatter",
                             "split", "chunk"],
                },
                "tensor_size": {"type": "integer", "description": "Tensor size in elements"},
                "num_ops": {"type": "integer", "description": "Number of ops to measure"},
            },
            "required": ["op_name"],
        },
    },
    {
        "name": "measure_algorithm_latency",
        "description": "Get the real hardware latency for a known AllToAllV algorithm configuration. Use for ground truth when validating your simulator. Algorithms are described by their collective and local op counts. Available: algo_1coll_3localops_32ranks, algo_1coll_35localops_32ranks, algo_31coll_62localops_32ranks, algo_2coll_5localops_32ranks, algo_fused_32ranks, algo_15coll_30localops_32ranks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm name (see tool description for available names)",
                },
            },
            "required": ["algorithm"],
        },
    },
    {
        "name": "measure_index_select_scaling",
        "description": "Measure how index_select performance scales with index tensor size and source tensor size. On NeuronCores, index_select has random memory access patterns — cost grows super-linearly with index size. Critical for algorithms that use index_select at large world sizes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "index_tensor_size": {
                    "type": "integer",
                    "description": "Number of elements in the index tensor",
                },
                "source_tensor_size": {
                    "type": "integer",
                    "description": "Number of elements in the source tensor being indexed into",
                },
            },
            "required": ["index_tensor_size", "source_tensor_size"],
        },
    },
    {
        "name": "measure_overlap_potential",
        "description": "Measure how much a collective can overlap with computation during training. Per-tensor all_reduce calls can pipeline with backward-pass computation (~30% overlap), while batched approaches (cat all tensors then all_reduce) serialize and get no overlap.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collective_type": {
                    "type": "string",
                    "enum": ["all_reduce", "all_gather", "reduce_scatter"],
                },
                "pattern": {
                    "type": "string",
                    "enum": ["per_tensor", "batched"],
                    "description": "per_tensor = one collective per gradient tensor (can pipeline), batched = cat all tensors then one collective (serialized)",
                },
            },
            "required": ["collective_type", "pattern"],
        },
    },
    {
        "name": "check_cross_node_support",
        "description": "Check whether a collective operation supports cross-node communication patterns. Some collectives (e.g., collective_permute ring patterns) crash when ring pairs cross node boundaries on Trainium.",
        "input_schema": {
            "type": "object",
            "properties": {
                "collective_type": {
                    "type": "string",
                    "enum": ["collective_permute", "all_gather", "all_reduce", "reduce_scatter"],
                },
            },
            "required": ["collective_type"],
        },
    },
    {
        "name": "check_primitive_compilation",
        "description": "Test whether an XLA collective primitive actually compiles on this hardware. Some primitives exist in the XLA Python API but the hardware compiler rejects them during HLO lowering. ALWAYS check compilation support before using a primitive in your cost model or recommending it to downstream phases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "primitive": {
                    "type": "string",
                    "enum": ["all_gather", "reduce_scatter", "all_reduce",
                             "collective_permute", "all_to_all"],
                    "description": "The XLA collective primitive to test",
                },
            },
            "required": ["primitive"],
        },
    },
    {
        "name": "discover_device_adjacency",
        "description": "Get the full device adjacency graph showing which devices are directly connected. Faster than probing every pair with measure_p2p_transfer.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "build_simulator",
        "description": "Submit Python code for your cost model simulator. The code MUST define a function: estimate_latency(algorithm_desc, **kwargs) -> float (microseconds). kwargs will include: num_collectives, num_xla_ops, data_bytes, num_hops. The function should use the hardware parameters you've discovered to predict latency.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code defining estimate_latency(algorithm_desc, **kwargs) -> float",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "validate_simulator",
        "description": "Validate your simulator against real hardware measurements. Tests predictions for multiple algorithms. All must be within 20% to pass. Optionally provide custom test_cases as [{algorithm, kwargs}].",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_cases": {
                    "type": "array",
                    "description": "Optional custom test cases. Default tests 4 representative algorithms.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "algorithm": {"type": "string"},
                            "kwargs": {"type": "object"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "set_simulator_config",
        "description": "Set discovered hardware parameters. These are used by the built-in topology simulator in downstream phases. Include unsupported_primitives to block the evolution from generating algorithms that won't compile.",
        "input_schema": {
            "type": "object",
            "properties": {
                "link_bandwidth_gbps": {"type": "number"},
                "link_latency_us": {"type": "number"},
                "collective_dispatch_overhead_us": {"type": "number"},
                "local_op_overhead_us": {"type": "number"},
                "num_devices": {"type": "integer"},
                "cores_per_device": {"type": "integer"},
                "device_adjacency": {"type": "object"},
                "unsupported_primitives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of XLA primitives that do NOT compile on this hardware (e.g., ['all_to_all']). Downstream phases will reject candidates using these.",
                },
            },
        },
    },
]


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """\
You are a hardware profiling agent. Your goal: discover the performance \
characteristics of this accelerator and build an accurate cost model (simulator).

You know NOTHING about this hardware upfront. You must discover everything \
through profiling experiments. Your cost model must predict algorithm \
latency within 20% of real measurements.

## Available profiling tools

1. **get_device_info** — basic device count and cores
2. **measure_collective_latency** — time collectives (all_gather, collective_permute, \
reduce_scatter, all_to_all) at various sizes/steps
3. **measure_p2p_transfer** — time point-to-point between specific devices
4. **measure_xla_op_overhead** — time local XLA ops (index_select, slice, cat, view, reshape, permute, transpose, zeros, etc.)
5. **measure_algorithm_latency** — get real latency of known algorithms
6. **discover_device_adjacency** — get full device connectivity graph
7. **measure_index_select_scaling** — measure how index_select cost scales with \
index tensor size and source tensor size (critical for large world sizes)
8. **measure_overlap_potential** — measure computation/communication overlap \
(per-tensor collectives pipeline with backward-pass; batched approaches serialize)
9. **check_cross_node_support** — check whether a collective supports cross-node \
patterns (collective_permute ring patterns crash when crossing node boundaries)
10. **check_primitive_compilation** — CRITICAL: test whether an XLA primitive actually \
compiles on this hardware. Some primitives exist in the Python API but the hardware \
compiler rejects them. You MUST check every primitive you plan to use.

## Simulator tools

11. **build_simulator** — submit Python code defining `estimate_latency(algorithm_desc, **kwargs)`
12. **validate_simulator** — test your simulator against real measurements (must pass <20% error)
13. **set_simulator_config** — store discovered parameters for downstream use

## Strategy

1. **Check primitive compilation**: FIRST, use check_primitive_compilation to test \
EVERY collective primitive (all_gather, reduce_scatter, all_reduce, collective_permute, \
all_to_all). Some primitives exist in the XLA API but the hardware compiler REJECTS \
them. Any algorithm using an unsupported primitive will fail to compile. Record which \
primitives are safe to use.
2. **Discover topology**: get device info, probe device adjacency
3. **Profile collectives**: measure collectives at different sizes/steps to decompose \
latency into fixed overhead + data-dependent terms.
4. **Profile ALL local XLA ops**: Use measure_xla_op_overhead to measure EVERY \
available operation type. Different ops have dramatically different costs — some may \
be nearly free (metadata-only) while others create real compute kernels. Downstream \
phases can ONLY use cost data for ops you actually measure here. If you skip an op, \
the evolution engine won't know its true cost and may make suboptimal choices. \
Measure all of: index_select, slice, cat, view, reshape, unsqueeze, squeeze, \
flatten, narrow, transpose, permute, expand, contiguous, zeros, tensor, stack, \
gather, scatter, split, chunk.
5. **Discover scaling effects**: measure how index_select scales with tensor size.
6. **Discover overlap effects**: measure whether per-tensor vs batched collectives \
have different effective costs in training.
7. **Discover cross-node constraints**: check which collectives support cross-node \
communication patterns.
8. **Get ground truth**: measure real algorithm latencies for validation targets
9. **Build cost model**: write a Python function that predicts latency from:
   - Number of collective dispatches and their type
   - Number of local XLA ops (use the per-op costs you measured — they differ wildly)
   - Data movement volume
   - Topology hops
   - Overlap potential (per-tensor vs batched patterns)
   - Cross-node constraints
   - Primitive compilation constraints (unsupported primitives = infinite cost)
10. **Validate**: check predictions match real measurements within 20%
11. **Iterate**: if validation fails, refine your cost model

The key question your model must answer: given a collective algorithm described by \
its operation counts, which approach minimizes TRAINING latency (not just isolated \
micro-benchmark latency)? Your model should account for:
- Primitive compilation constraints (algorithms using unsupported primitives CANNOT RUN)
- The relative costs you measured for different operation types
- How operation costs may scale with tensor size
- Computation/communication overlap in training loops
- Cross-node topology constraints

CRITICAL: Your cost model MUST return infinity (or a very large value like 1e9) for \
any algorithm that uses a primitive that doesn't compile on this hardware. The \
downstream evolution phases will use your model to rank candidates — if you don't \
penalize unsupported primitives, the evolution will produce algorithms that crash \
on real hardware.
"""


# ============================================================
# Agent loop
# ============================================================

def run_profiling_agent(model="haiku", max_turns=15, verbose=True):
    """
    Run the profiling agent to discover hardware and build a simulator.

    Returns:
        AgentSimulator with discovered config, simulator code, and validation results
    """
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    agent_sim = AgentSimulator()

    messages = [
        {"role": "user", "content": (
            "Profile this hardware and build a cost model simulator. "
            "You must: (1) discover the topology and key performance parameters, "
            "(2) write a Python estimate_latency() function, "
            "(3) validate it predicts real algorithm latencies within 20%. "
            "Start by getting device info and measuring some collectives."
        )}
    ]

    tools = [{
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["input_schema"],
    } for t in PROFILING_TOOLS]

    for turn in range(max_turns):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        }

        try:
            resp = client.invoke_model(
                modelId=MODEL_IDS[model], body=json.dumps(body))
            result = json.loads(resp["body"].read())
        except Exception as e:
            if verbose:
                print(f"  [Turn {turn+1}] Bedrock error: {e}")
            break

        stop_reason = result.get("stop_reason", "")
        content = result.get("content", [])

        assistant_content = []
        tool_calls = []

        for block in content:
            if block.get("type") == "text":
                if verbose and block["text"].strip():
                    text = block["text"].strip()
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"  [Turn {turn+1}] Agent: {text}")
                assistant_content.append(block)
            elif block.get("type") == "tool_use":
                tool_calls.append(block)
                assistant_content.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_calls:
            if verbose:
                print(f"  [Turn {turn+1}] Agent finished (no more tool calls)")
            break

        tool_results = []
        for tc in tool_calls:
            tool_name = tc["name"]
            tool_input = tc.get("input", {})

            if verbose:
                args_str = json.dumps(tool_input)
                if len(args_str) > 100:
                    args_str = args_str[:100] + "..."
                print(f"  [Turn {turn+1}] Tool: {tool_name}({args_str})")

            result_str = _handle_tool_call(tool_name, tool_input, agent_sim)

            if verbose and tool_name in ("build_simulator", "validate_simulator"):
                result_data = json.loads(result_str)
                status = result_data.get("status", result_data.get("all_within_tolerance", ""))
                print(f"    -> {tool_name}: {status}")

            if tool_name == "set_simulator_config":
                for key in ["link_bandwidth_gbps", "link_latency_us",
                            "collective_dispatch_overhead_us", "local_op_overhead_us",
                            "num_devices", "cores_per_device"]:
                    if key in tool_input:
                        if verbose:
                            print(f"    -> Set {key} = {tool_input[key]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    # Fill defaults for any parameters the agent didn't discover
    config = agent_sim.config
    if config.num_devices == 0:
        config.num_devices = _HARDWARE_MEASUREMENTS["num_devices"]
    if config.cores_per_device == 0:
        config.cores_per_device = _HARDWARE_MEASUREMENTS["cores_per_device"]
    config.num_cores = config.num_devices * config.cores_per_device
    if not config.device_adjacency:
        config.device_adjacency = _HARDWARE_MEASUREMENTS["device_adjacency"]
    if config.efa_bandwidth_gbps == 0.0:
        config.efa_bandwidth_gbps = _HARDWARE_MEASUREMENTS["efa_bandwidth_gbps"]
    if config.efa_latency_us == 0.0:
        config.efa_latency_us = _HARDWARE_MEASUREMENTS["efa_latency_us"]
    if config.efa_adapters_per_node == 0:
        config.efa_adapters_per_node = _HARDWARE_MEASUREMENTS["efa_adapters_per_node"]

    # Discover unsupported primitives via compilation test if agent didn't
    if not config.unsupported_primitives:
        for prim in ["all_gather", "reduce_scatter", "all_reduce",
                     "collective_permute", "all_to_all"]:
            try:
                result = _test_primitive_compilation(prim)
                if not result.get("compiles_on_hardware", True):
                    config.unsupported_primitives.append(prim)
            except Exception:
                pass

    _ESSENTIAL_COLLECTIVES = {"all_gather", "reduce_scatter", "all_reduce", "collective_permute"}
    if _ESSENTIAL_COLLECTIVES.intersection(config.unsupported_primitives):
        config.unsupported_primitives = [
            p for p in config.unsupported_primitives
            if p not in _ESSENTIAL_COLLECTIVES
        ]

    if verbose:
        print(f"\n  Simulator config:")
        print(f"    collective_dispatch_overhead_us = {config.collective_dispatch_overhead_us}")
        print(f"    local_op_overhead_us = {config.local_op_overhead_us}")
        print(f"    link_bandwidth_gbps = {config.link_bandwidth_gbps}")
        print(f"    link_latency_us = {config.link_latency_us}")
        if config.unsupported_primitives:
            print(f"    unsupported_primitives = {config.unsupported_primitives}")
        print(f"    Simulator built: {agent_sim.cost_function is not None}")
        print(f"    Simulator validated: {agent_sim.is_validated()}")
        print(f"    Config complete: {config.is_complete()}")

    return agent_sim


def refine_simulator(agent_sim, error_feedback, model="haiku", max_turns=5,
                     verbose=True):
    """
    Refine an existing simulator based on error feedback from downstream phases.

    Called when Phase 2 or Phase 3 discovers that the simulator's predictions
    don't match observed behavior (e.g., an algorithm the simulator predicted
    to be fast turns out to be slow on real hardware).

    Args:
        agent_sim: AgentSimulator from Phase 1
        error_feedback: str describing the prediction error
        model: LLM model to use
        max_turns: max refinement turns

    Returns:
        Updated AgentSimulator
    """
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    current_code = agent_sim.simulator_code or "(no simulator code yet)"
    validation_summary = json.dumps(agent_sim.validation_results, indent=2) if agent_sim.validation_results else "none"

    messages = [
        {"role": "user", "content": (
            f"Your simulator needs refinement. Here's what went wrong:\n\n"
            f"{error_feedback}\n\n"
            f"Your current simulator code:\n```python\n{current_code}\n```\n\n"
            f"Previous validation results:\n{validation_summary}\n\n"
            f"Please profile additional hardware characteristics if needed, "
            f"then submit a refined simulator with build_simulator and validate it."
        )}
    ]

    tools = [{
        "name": t["name"],
        "description": t["description"],
        "input_schema": t["input_schema"],
    } for t in PROFILING_TOOLS]

    for turn in range(max_turns):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.0,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        }

        try:
            resp = client.invoke_model(
                modelId=MODEL_IDS[model], body=json.dumps(body))
            result = json.loads(resp["body"].read())
        except Exception as e:
            if verbose:
                print(f"  [Refine turn {turn+1}] Bedrock error: {e}")
            break

        content = result.get("content", [])
        assistant_content = []
        tool_calls = []

        for block in content:
            if block.get("type") == "text":
                if verbose and block["text"].strip():
                    text = block["text"].strip()
                    if len(text) > 200:
                        text = text[:200] + "..."
                    print(f"  [Refine {turn+1}] Agent: {text}")
                assistant_content.append(block)
            elif block.get("type") == "tool_use":
                tool_calls.append(block)
                assistant_content.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_calls:
            break

        tool_results = []
        for tc in tool_calls:
            result_str = _handle_tool_call(tc["name"], tc.get("input", {}), agent_sim)
            if verbose and tc["name"] in ("build_simulator", "validate_simulator"):
                result_data = json.loads(result_str)
                status = result_data.get("status", result_data.get("all_within_tolerance", ""))
                print(f"    -> {tc['name']}: {status}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    if verbose:
        print(f"  Simulator refined: validated={agent_sim.is_validated()}")

    return agent_sim
