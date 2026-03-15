#=
    @graphbreak function handling.

    Called from inside @graphbreak-wrapped functions. Behavior depends on the current
    capture context mode:

    :off          → just run the function body (no-op wrapper)
    :capturing    → end segment, store closure, run body eagerly, start next segment
    :replaying    → launch preceding cached segment, run body eagerly with current args
    :recapturing  → end segment, update/instantiate cached exec, launch it, run body
                    eagerly with current args, start next segment capture
=#

"""
    _at_break!(f)

Handle a graph break. `f` is a zero-argument closure over the function body and its
arguments, created by the `@graphbreak` macro wrapper.
"""
function _at_break!(f)
    ctx = _CTX[]
    mode = ctx.mode

    if mode === :capturing
        _break_capture!(ctx, f)
    elseif mode === :replaying
        _break_replay!(ctx, f)
    elseif mode === :recapturing
        _break_recapture!(ctx, f)
    else
        # :off — normal execution
        return f()
    end
end

# --- :capturing (used by @unsafe_scaptured first capture) ---

function _break_capture!(ctx::CaptureContext, f)
    stream = ctx.stream
    cache = ctx.cache

    # End current segment capture
    graph = _end_capture(stream)
    exec = _instantiate(graph)
    push!(cache.graphs, graph)
    push!(cache.execs, exec)
    push!(cache.break_closures, f)

    # Skip break work during capture — it would run on stale data anyway.
    # The stored closure will run during replay with correct data.

    # Start next segment capture
    ctx.segment += 1
    _begin_capture(stream)
    return nothing
end

# --- :replaying (used by @unsafe_scaptured subsequent iterations) ---

function _break_replay!(ctx::CaptureContext, f)
    cache = ctx.cache
    seg = ctx.segment

    # Launch preceding graph segment
    _launch(cache.execs[seg], ctx.stream)

    # Run break work with CURRENT arguments (from stored closure)
    result = cache.break_closures[seg]()

    ctx.segment += 1
    return result
end

# --- :recapturing (used by @scaptured every iteration) ---

function _break_recapture!(ctx::CaptureContext, f)
    stream = ctx.stream
    cache = ctx.cache
    seg = ctx.segment

    # End current segment capture → get graph of recorded kernels
    graph = _end_capture(stream)

    # Update or create the cached exec for this segment
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

    # Launch the (updated) segment
    _launch(cache.execs[seg], stream)

    # Run break work eagerly with CURRENT arguments
    result = f()

    # Resume capture for next segment
    ctx.segment += 1
    _begin_capture(stream)
    return result
end
