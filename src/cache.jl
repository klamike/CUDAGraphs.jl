"""
    SegmentedGraphCache

Stores captured graph segments and (for `@unsafe_scaptured`) break closures.

Used internally by `@scaptured` (one per call site, auto-managed) and explicitly
by `@unsafe_scaptured` (user-created, user-invalidated).
"""
mutable struct SegmentedGraphCache
    execs::Vector{CUDA.CUgraphExec}    # raw exec handles (one per segment)
    graphs::Vector{CUDA.CUgraph}       # raw graph handles (for cleanup)
    break_closures::Vector{Any}        # break work closures (used by @unsafe_scaptured replay)
    n_segments::Int                     # number of captured segments
    valid::Bool
end

function SegmentedGraphCache()
    SegmentedGraphCache(
        CUDA.CUgraphExec[], CUDA.CUgraph[], Any[], 0, false,
    )
end

"""
    invalidate!(cache::SegmentedGraphCache)

Destroy all cached graph segments and mark the cache as invalid.
The next `@unsafe_scaptured` call will re-capture.
"""
function invalidate!(cache::SegmentedGraphCache)
    for exec in cache.execs
        CUDA.cuGraphExecDestroy(exec)
    end
    for graph in cache.graphs
        CUDA.cuGraphDestroy(graph)
    end
    empty!(cache.execs)
    empty!(cache.graphs)
    empty!(cache.break_closures)
    cache.n_segments = 0
    cache.valid = false
    return
end
