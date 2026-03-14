#=
    Low-level CUDA graph API wrappers using raw handles.

    We use raw CUgraph/CUgraphExec handles instead of CUDA.jl's CuGraph/CuGraphExec
    types because CuGraph's inner constructor is only accessible from within its
    struct body (via capture()). Since we need to construct graphs from
    cuStreamEndCapture handles outside of capture(), we work at the raw handle level
    and manage lifecycle manually via SegmentedGraphCache.
=#

function _begin_capture(stream::CUDA.CuStream)
    CUDA.cuStreamBeginCapture_v2(stream, CUDA.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
end

function _end_capture(stream::CUDA.CuStream)
    graph_ref = Ref{CUDA.CUgraph}()
    err = CUDA.unchecked_cuStreamEndCapture(stream, graph_ref)
    err != CUDA.CUDA_SUCCESS && throw(CUDA.CuError(err))
    return graph_ref[]
end

function _instantiate(graph::CUDA.CUgraph)
    exec_ref = Ref{CUDA.CUgraphExec}()
    CUDA.cuGraphInstantiateWithFlags(exec_ref, graph, UInt64(0))
    return exec_ref[]
end

function _launch(exec::CUDA.CUgraphExec, stream::CUDA.CuStream)
    CUDA.cuGraphLaunch(exec, stream)
end

"""
    _try_update(exec, graph) -> Bool

Try to update an existing executable graph with a new graph.
Returns true if the update succeeded (same topology), false otherwise.
"""
function _try_update(exec::CUDA.CUgraphExec, graph::CUDA.CUgraph)
    error_node = Ref{CUDA.CUgraphNode}()
    update_result = Ref{CUDA.CUgraphExecUpdateResult}()
    CUDA.cuGraphExecUpdate(exec, graph, error_node, update_result)
    return update_result[] == CUDA.GRAPH_EXEC_UPDATE_SUCCESS
end
