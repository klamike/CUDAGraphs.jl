#=
    @scaptured implementation — safe, automatic mode.

    Re-captures every iteration. Uses cuGraphExecUpdate to cheaply update cached
    executables when the graph topology hasn't changed. Break functions run with
    current arguments every iteration — no closures, no invalidation needed.

    Cost per iteration: (N+1) begin/end capture + (N+1) cuGraphExecUpdate + (N+1)
    graph launches + N break calls. For N=3 breaks that's ~4x the overhead of a
    single CUDA.@captured, negligible when GPU work dominates.
=#

function _run_scaptured!(f, cache::SegmentedGraphCache)
    stream = CUDA.stream()
    ctx = _CTX[]
    @assert ctx.mode === :off "nested @scaptured / @unsafe_scaptured not supported"

    ctx.mode = :recapturing
    ctx.cache = cache
    ctx.segment = 1
    ctx.stream = stream

    gc = GC.enable(false)
    try
        _begin_capture(stream)
        f()  # user block: kernels recorded, breaks update+launch+execute
        # End final segment
        graph = _end_capture(stream)
        seg = ctx.segment
        if seg <= length(cache.execs)
            if !_try_update(cache.execs[seg], graph)
                CUDA.cuGraphExecDestroy(cache.execs[seg])
                cache.execs[seg] = _instantiate(graph)
            end
            CUDA.cuGraphDestroy(graph)
        else
            push!(cache.execs, _instantiate(graph))
            push!(cache.graphs, graph)
        end
        _launch(cache.execs[seg], stream)
        cache.n_segments = seg
        cache.valid = true
    catch
        # Capture failed, probably JIT compilation. Next call should capture successfully.
        try; _end_capture(stream); catch; end
        invalidate!(cache)
        ctx.mode = :off
        GC.enable(gc)
        f()
        return
    finally
        ctx.mode = :off
        GC.enable(gc)
    end
end
