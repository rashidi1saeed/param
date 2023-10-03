import json
import logging
import sys

import networkx as nx
from networkx.algorithms import isomorphism
from utility import (
    load_execution_trace_file,
    read_dictionary_from_json_file,
)


# Increase recursion limit
sys.setrecursionlimit(10**6)

logger = logging.getLogger()

execution_trace_process_annotation = "[pytorch|profiler|execution_trace|process]"
execution_trace_thread_annotation = "[pytorch|profiler|execution_trace|thread]"


# Add and sort ET nodes from the execution trace
def collect_nodes(node):
    def traverse(node):
        nonlocal nodes
        nodes.append(node)
        for child in node.children:
            traverse(child)

    nodes = []
    traverse(node)
    sorted_nodes = sorted(nodes, key=lambda x: x.id)
    return sorted_nodes


class Kineto_node:
    def __init__(self, name, start, end, id):
        self.name = name
        self.start = start
        self.end = end
        self.id = id
        self.children = []


# Function to transform your self-defined tree to a directed graph with max depth
def transform_to_graph_depth(node, max_depth=100):
    graph = nx.DiGraph()
    add_node_to_graph_depth(node, graph, max_depth, 1)
    return graph


# Helper function to recursively add nodes and edges to the graph with max depth
def add_node_to_graph_depth(node, graph, max_depth, cur_depth):
    graph.add_node(node.id, label=node.name)
    if cur_depth == max_depth:
        return
    for child in node.children:
        graph.add_node(node.id, label=node.name)
        add_node_to_graph_depth(child, graph, max_depth, cur_depth + 1)


# Custom node comparison function for edit distance
def node_compare(n1, n2):
    return n1 == n2


# Find the segment that has a length closest to the target
def find_closest_segment(segs, target_length):
    closest_length = float("inf")
    closest_seg = None

    for seg in segs:
        length_difference = abs(len(seg) - target_length)
        if length_difference < closest_length:
            closest_length = length_difference
            closest_seg = seg

    return closest_seg

def find_parent_cpu_op(kineto_gpu_event, kineto_et_events, kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events):
    ts=-1
    method=""
    if kineto_gpu_event["args"]["External id"] in kineto_cpu_launch_kernel_events.keys():
        ts=kineto_cpu_launch_kernel_events[kineto_gpu_event["args"]["External id"]]["ts"]+kineto_cpu_launch_kernel_events[kineto_gpu_event["args"]["External id"]]["dur"]
        method="cuda launch"
    elif kineto_gpu_event["args"]["External id"] in kineto_ac2g_s_events.keys():
        ts=kineto_ac2g_s_events[kineto_gpu_event["args"]["External id"]]["ts"]
        method="ac2g_s"
    elif kineto_gpu_event["args"]["External id"] in kineto_ac2g_f_events.keys():
        ts=kineto_ac2g_f_events[kineto_gpu_event["args"]["External id"]]["ts"]    
        method="ac2g_f"
    assert ts!=-1

    kineto_gpu_event["ts"]=ts
    closest_start=0
    parent_cpu_op={}
    for event in kineto_et_events:
        if "cat" in event and event["ts"]<kineto_gpu_event["ts"] and (event["ts"]+event["dur"])>kineto_gpu_event["ts"] and event["ts"]>closest_start:
            closest_start=event["ts"]
            parent_cpu_op=event
    if not parent_cpu_op:
        print("Warning! the parent cpu_op for the following gpu_op is not found and hence it is discarded. gpu_op name: "+str(kineto_gpu_event["name"])+", ts: "+str(kineto_gpu_event["ts"])+", external_id: "+str(kineto_gpu_event["args"]["External id"]))
    #assert parent_cpu_op and parent_cpu_op["ts"]+parent_cpu_op["dur"]>kineto_gpu_event["ts"]
    return parent_cpu_op

# Extract operator info from raw traces
def trace_analysis(et_file, kineto_file, annotation="DataLoader"):
    et = load_execution_trace_file(et_file)

    nodes = et.get_nodes()

    # Root node of execution trace is 1-based
    et_nodes = collect_nodes(nodes[1])

    logger.info(f"Number of original ops in execution trace: {len(et_nodes)}")

    kineto_trace_events = read_dictionary_from_json_file(kineto_file)["traceEvents"]

    sorted_kineto_trace_events = sorted(kineto_trace_events, key=lambda kv: kv["ts"])

    kineto_et_events = [
        event
        for event in sorted_kineto_trace_events
        if "cat" in event and "ProfilerStep" not in event["name"]
        and (event["cat"] == "cpu_op" or event["cat"] == "user_annotation")
    ]

    kineto_ac2g_s_events = {
        event["id"]:event
        for event in sorted_kineto_trace_events
        if "cat" in event and "ProfilerStep" not in event["name"]
        and (event["cat"] == "ac2g" and event["ph"] == "s")
    }
    
    kineto_ac2g_f_events = {
        event["id"]:event
        for event in sorted_kineto_trace_events
        if "cat" in event and "ProfilerStep" not in event["name"]
        and (event["cat"] == "ac2g" and event["ph"] == "f")
    }
    
    kineto_cpu_launch_kernel_events = {
        event["args"]["External id"]:event
        for event in sorted_kineto_trace_events
        if "cat" in event 
        and (event["cat"] == "cuda_runtime" and (event["name"] == "cudaLaunchKernel" or event["name"] == "cudaMemcpyAsync"))
    }

    kineto_gpu_events = [
        event
        for event in sorted_kineto_trace_events
        if "cat" in event and "ProfilerStep" not in event["name"] 
        and (event["cat"] == "kernel" or event["cat"] == "gpu_memcpy")
    ]

    kineto_iteration_latencies = [
        iteration["dur"]
        for iteration in sorted_kineto_trace_events
        if "ProfilerStep" in iteration["name"]
    ]
    average_iteration_latency = 0
    if len(kineto_iteration_latencies) > 0:
        average_iteration_latency=sum(kineto_iteration_latencies)/len(kineto_iteration_latencies)

    kineto_et_segs = []
    kineto_et_seg = []

    # The choice below normally does not matter for approximate match since we rely on the isomorphism of
    # the graphs, but for exact match we will use the execution order and then we should be careful

    # Assume that an iteration ends with the specified annotation
    end_time = -1
    for event in kineto_et_events:
        if end_time > 0 and event["ts"] >= end_time:
            kineto_et_segs.append(kineto_et_seg)
            kineto_et_seg = []
            end_time = -1

        if annotation in event["name"]:
            kineto_et_seg.append(event)
            end_time = event["ts"] + event["dur"]
        else:
            kineto_et_seg.append(event)

    # # Assume that an iteration starts with the specified annotation
    # for event in kineto_et_events:
    #     if annotation in event['name']:
    #         kineto_et_segs.append(kineto_et_seg)
    #         kineto_et_seg = [event]
    #     else:
    #         kineto_et_seg.append(event)

    # In case of kineto only contains one iteration or the provided annotation is wrong, use the whole trace directly,
    # otherwise find the iteration in kineto trace with the closest #ops to ET (usually ET has 3 additional annotation ops for processes/threads)
    if kineto_et_segs:
        kineto_et_events = find_closest_segment(kineto_et_segs, len(et_nodes) - 3)
    
    if average_iteration_latency > 0:
        logger.info(f"Number of original cpu ops in kineto trace: {len(kineto_et_events)}, Number of original gpu ops in kineto trace: {len(kineto_gpu_events)}, average iteration latency: {average_iteration_latency}")
    else:
        logger.info(f"Number of original cpu ops in kineto trace: {len(kineto_et_events)}, Number of original gpu ops in kineto trace: {len(kineto_gpu_events)}")

    return et_nodes, kineto_et_events,kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events,kineto_gpu_events

def exist(name,kineto_et_events,i):
    MAX_DISTANCE=0
    distance=0
    while distance<=MAX_DISTANCE and distance+i<len(kineto_et_events) or i-distance>=0:
        if distance+i<len(kineto_et_events) and name==kineto_et_events[distance+i]["name"]:
            return True,kineto_et_events[distance+i]
        elif i-distance>=0 and name==kineto_et_events[i-distance]["name"]:
            return True,kineto_et_events[i-distance]
        distance+=1
    return False,kineto_et_events[i]
    
def exact_match(kineto_et_events,kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events,kineto_gpu_events, et_nodes):
    # Since kineto trace is missing the annotations for processes/threads, we add them back to match with ET
    kineto_event_per_thread = {}

    process_end_time = -1
    for i in range(len(kineto_et_events)):
        event = kineto_et_events[i]
        if event["tid"] not in kineto_event_per_thread:
            kineto_event_per_thread[event["tid"]] = {}
            kineto_event_per_thread[event["tid"]]["ts"] = event["ts"]
            kineto_event_per_thread[event["tid"]]["end_ts"] = event["ts"] + event["dur"]
            kineto_event_per_thread[event["tid"]]["index"] = i
        else:
            kineto_event_per_thread[event["tid"]]["end_ts"] = max(
                kineto_event_per_thread[event["tid"]]["end_ts"],
                event["ts"] + event["dur"],
            )
        process_end_time = max(process_end_time, event["ts"] + event["dur"])

    process_event = {
        "name": execution_trace_process_annotation,
        "ts": kineto_et_events[0]["ts"],
        "dur": process_end_time - kineto_et_events[0]["ts"],
    }

    kineto_et_events.insert(0, process_event)

    sorted_threads = dict(
        sorted(kineto_event_per_thread.items(), key=lambda x: x[1]["index"])
    )

    for index, (tid, thread_info) in enumerate(sorted_threads.items()):
        thread_event = {
            "name": execution_trace_thread_annotation,
            "ts": thread_info["ts"],
            "dur": thread_info["end_ts"] - thread_info["ts"],
        }
        # Be careful of the insertion position, note that we already inserted process event
        kineto_et_events.insert(index + 1 + thread_info["index"], thread_event)

    # Duration of ET nodes
    et_enhanced_duration = {}
    # Timestamp of ET nodes
    et_enhanced_timestamp = {}

    gpu_kernels_per_cpu_event_id={}
    gpu_kernels_per_cpu_event_idx={}
    for gpu_event in kineto_gpu_events:
        parent_cpu_event=find_parent_cpu_op(gpu_event,kineto_et_events,kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events)
        if not parent_cpu_event:
            continue
        assert "Ev Idx" in parent_cpu_event["args"]
        if parent_cpu_event["args"]["Ev Idx"] not in gpu_kernels_per_cpu_event_idx:
            gpu_kernels_per_cpu_event_idx[parent_cpu_event["args"]["Ev Idx"]]=[gpu_event]
        else:
            gpu_kernels_per_cpu_event_idx[parent_cpu_event["args"]["Ev Idx"]].append(gpu_event)
    # Link kineto trace and execution trace
    if len(kineto_et_events) == len(et_nodes):
        for i in range(len(et_nodes)):
            et_node = et_nodes[i]
            name_exist,kineto_et_event = exist(et_node.name,kineto_et_events,i)
            if (
                name_exist
                or (
                    "iteration#" in et_node.name
                    and "iteration#" in kineto_et_event["name"]
                )
                or et_node.name.replace("execution_graph", "execution_trace")
                == kineto_et_event["name"]
            ):
                et_enhanced_duration[et_node.id] = kineto_et_event["dur"]
                et_enhanced_timestamp[et_node.id] = kineto_et_event["ts"]
                if "args" in kineto_et_event and "Ev Idx" in kineto_et_event["args"] and kineto_et_event["args"]["Ev Idx"] in gpu_kernels_per_cpu_event_idx:
                    gpu_kernels_per_cpu_event_id[et_node.id]=gpu_kernels_per_cpu_event_idx[kineto_et_event["args"]["Ev Idx"]]
            else:
                logger.info("Op mismatch between kineto and execution trace ( et size = "+str(len(et_nodes))+", kineto size: "+str(len(kineto_et_events))+" ):")
                logger.info(
                    f'Op index: {i}, kineto op name: {kineto_et_event["name"]}, kineto op timestamp: {kineto_et_event["ts"]}, '
                    f"execution trace op name: {et_node.name}, execution trace op id: {et_node.id}"
                )
                for i in range(len(kineto_et_events)):
                    kineto_et_event = kineto_et_events[i]
                    et_node = et_nodes[i]    
                    logger.info( "Index: "+str(i)+", et name: " + et_node.name+ ", kineto name: " + kineto_et_event["name"])
                   
                exit(0)
    else:
        logger.info("Ops count mismatch between kineto and execution trace ( et size = "+str(len(et_nodes))+", kineto size: "+str(len(kineto_et_events))+" )")

    return et_enhanced_duration,et_enhanced_timestamp,gpu_kernels_per_cpu_event_id


def approximate_match(kineto_et_events, et_nodes):
    # Since kineto trace is missing the annotations for processes/threads, we add them back to match with ET
    kineto_event_per_thread = {}

    # Mapping node id to the corresponding node
    kineto_nodes_mapping = {}

    for event in kineto_et_events:
        if event["tid"] not in kineto_event_per_thread:
            kineto_event_per_thread[event["tid"]] = []
        kineto_event_per_thread[event["tid"]].append(event)

    start_time = kineto_et_events[0]["ts"]
    end_time = -1
    for event in kineto_et_events:
        end_time = max(end_time, event["ts"] + event["dur"])
    process_node = Kineto_node(
        execution_trace_process_annotation, start_time, end_time, 0
    )
    kineto_nodes_mapping[0] = process_node

    cnt = 1
    for thread in kineto_event_per_thread:
        start_time = kineto_event_per_thread[thread][0]["ts"]
        end_time = -1
        for event in kineto_event_per_thread[thread]:
            end_time = max(end_time, event["ts"] + event["dur"])

        thread_node = Kineto_node(
            execution_trace_thread_annotation, start_time, end_time, cnt
        )
        kineto_nodes_mapping[cnt] = thread_node
        cnt += 1

        process_node.children.append(thread_node)

        kineto_nodes = [thread_node]
        for event in kineto_event_per_thread[thread]:
            if event["ts"] < kineto_nodes[-1].end:
                tmp = Kineto_node(
                    event["name"], event["ts"], event["ts"] + event["dur"], cnt
                )
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)
            else:
                while len(kineto_nodes)>0 and kineto_nodes[-1].end <= event["ts"]:
                    kineto_nodes.pop()
                tmp = Kineto_node(
                    event["name"], event["ts"], event["ts"] + event["dur"], cnt
                )
                kineto_nodes_mapping[cnt] = tmp
                cnt += 1
                if len(kineto_nodes)>0:
                    kineto_nodes[-1].children.append(tmp)
                kineto_nodes.append(tmp)

    # Max call stack depth when building the tree, the deeper the more accurate but takes longer time
    depth = 10

    # Build a tree from the kineto trace
    kineto_graph = transform_to_graph_depth(process_node, depth)
    logger.info(f"Kineto tree nodes number: {len(kineto_graph.nodes)}")

    # Build a tree from the execution trace
    et_graph = transform_to_graph_depth(et_nodes[0], depth)
    logger.info(f"ET tree nodes number: {len(et_graph.nodes)}")

    # Create the GraphMatcher
    GM = isomorphism.GraphMatcher(kineto_graph, et_graph)

    # Duration of ET nodes
    et_enhanced_duration = {}

    if GM.is_isomorphic():
        mapping = GM.mapping
        logger.info("Graphs are isomorphic")
        for kineto_id, et_id in mapping.items():
            et_enhanced_duration[et_id] = (
                kineto_nodes_mapping[kineto_id].end
                - kineto_nodes_mapping[kineto_id].start
            )
    else:
        logger.info("Graphs are not isomorphic")

        # # Compute the edit distance using the graph_edit_distance function with node comparison
        # paths, cost = nx.graph_edit_distance(kineto_graph, et_graph, node_compare)
        # logger.info(f"Tree edit distance: {cost}")

        # The problem of finding the exact Graph Edit Distance (GED) is NP-hard so it is often slow
        # and below is a sub-optimal approach

        edit_distance_generator = nx.optimize_graph_edit_distance(
            kineto_graph, et_graph, node_compare
        )
        cost = next(edit_distance_generator)

        paths_generator = nx.optimize_edit_paths(kineto_graph, et_graph, node_compare)
        node_edits, _, cost = next(paths_generator)

        logger.info(f"Sub-optimal tree edit distance: {cost}")

        for kineto_id, et_id in node_edits:
            if kineto_id is not None and et_id is not None:
                et_enhanced_duration[et_id] = (
                    kineto_nodes_mapping[kineto_id].end
                    - kineto_nodes_mapping[kineto_id].start
                )

    return et_enhanced_duration

def assign_ids(total_assigned_ids,assigned_ids,id):
    orig_id=id
    while True:
        if id in total_assigned_ids:
            id+=1
        else:
            total_assigned_ids.append(id)
            if orig_id not in assigned_ids.keys():
                assigned_ids[orig_id]=id
            return id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Link kineto trace with execution trace"
    )
    parser.add_argument(
        "--et-file", type=str, required=True, help="Path to the execution trace"
    )
    parser.add_argument(
        "--kineto-file", type=str, required=True, help="Path to the kineto trace"
    )
    parser.add_argument(
        "--annotation",
        default="DataLoader",
        type=str,
        help="Operator name to help slice multiple iterations in trace",
    )
    parser.add_argument(
        "--exact-match",
        default=False,
        action="store_true",
        help="Whether to match the traces exactly",
    )
    parser.add_argument("--log-level", default="INFO", help="Log output verbosity.")

    args = parser.parse_args()

    logger.setLevel(args.log_level)
    
    gpu_kernels_per_cpu_event_id={}
    et_nodes, kineto_et_events, kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events,kineto_gpu_events = trace_analysis(
        args.et_file, args.kineto_file, args.annotation
    )

    if args.exact_match:
        et_enhanced_duration,et_enhanced_timestamp, gpu_kernels_per_cpu_event_id = exact_match(kineto_et_events,kineto_ac2g_s_events,kineto_ac2g_f_events,kineto_cpu_launch_kernel_events,kineto_gpu_events, et_nodes)
    else:
        logger.info("This script works only with the exact match mode, please add the --exact-match flag to your script and run it again")
        assert False
        #et_enhanced_duration = approximate_match(kineto_et_events, et_nodes)

    # If linking works, add duration time to each ET node and dump as ET_plus
    if et_enhanced_duration:
        assigned_ids={}
        total_assigned_ids=[]
        with open(args.et_file, "r") as f:
            et = json.load(f)
            for node in et["nodes"]:
                if "cat" in node.keys():
                    break
                node_id=node["id"]
                node["id"] = assign_ids(total_assigned_ids,assigned_ids,node_id)
                if node_id in et_enhanced_duration:
                    node["dur"] = et_enhanced_duration[node_id]
                    node["ts"] = et_enhanced_timestamp[node_id]
                if node_id in gpu_kernels_per_cpu_event_id:
                    gpu_kernels_per_cpu_event_id[node_id]=sorted(gpu_kernels_per_cpu_event_id[node_id], key=lambda kv: kv["ts"])
                    gpu_nodes=gpu_kernels_per_cpu_event_id[node_id]
                    for gpu_node in gpu_nodes:
                        #if gpu_node["args"]["External id"]==41900:
                        #    print("before dump ts: "+str(gpu_node["ts"])+", dur: "+str(gpu_node["dur"]))
                        gpu_node["parent"]=node["id"]
                        #print("parent id: "+str(node["id"]))
                        gpu_node["id"]=assign_ids(total_assigned_ids,assigned_ids,node_id)
                        #print("node id: "+ str(gpu_node["id"]))
                        gpu_node["inputs"]=node["inputs"]
                        gpu_node["input_shapes"]=node["input_shapes"]
                        gpu_node["input_types"]=node["input_types"]
                        gpu_node["outputs"]=node["outputs"]
                        gpu_node["output_shapes"]=node["output_shapes"]
                        gpu_node["output_types"]=node["output_types"]
                        et["nodes"].append(gpu_node)
                    gpu_kernels_per_cpu_event_id.pop(node_id)
            for node in et["nodes"]:
                if "cat" not in node.keys():
                    node["parent"]=assigned_ids[node["parent"]]
        et_plus_file = args.et_file.replace(".json", "_plus.json")
        logger.info(f"Enhanced execution trace dumped to {et_plus_file}.")
        with open(et_plus_file, "w") as f:
            json.dump(et, f, indent=4)
