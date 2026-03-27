#=
    @unsafe_scaptured implementation — fast, manual mode.

    First call: captures all segments + stores break closures.
    Subsequent calls: replays cached segments + calls stored closures.
    User must call invalidate!(cache) when break function arguments change.

    Cost per iteration (after first): (N+1) graph launches + N closure calls.
    Minimal CPU overhead — no re-capture, no cuGraphExecUpdate.
=#

function _run_unsafe_scaptured!(f, cache::SegmentedGraphCache)
    if cache.valid
        _unsafe_replay!(cache)
    else
        _unsafe_capture_and_replay!(f, cache)
    end
end

function _unsafe_capture_and_replay!(f, cache::SegmentedGraphCache)
    stream = CUDA.stream()
    ctx = _CTX[]
    @assert ctx.mode === :off "nested @scaptured / @unsafe_scaptured not supported"

    ctx.mode = :capturing
    ctx.cache = cache
    ctx.segment = 1
    ctx.stream = stream

    gc = GC.enable(false)
    ok = true
    try
        _begin_capture(stream)
        f()  # user block: kernels recorded, breaks end+store+run+begin
        # End final segment
        graph = _end_capture(stream)
        exec = _instantiate(graph)
        push!(cache.graphs, graph)
        push!(cache.execs, exec)
        cache.n_segments = ctx.segment
        cache.valid = true
    catch err
        ok = false
        bt = catch_backtrace()
        _report_capture_failure(:unsafe_scaptured, err, bt)
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

    # Replay to get correct first-iteration results
    # (capture-time kernel work was recorded but not executed)
    _unsafe_replay!(cache)
end

function _unsafe_replay!(cache::SegmentedGraphCache)
    stream = CUDA.stream()
    n = cache.n_segments
    for seg in 1:n
        _launch(cache.execs[seg], stream)
        if seg < n
            cache.break_closures[seg]()  # stream-ordered after graph
        end
    end
end
