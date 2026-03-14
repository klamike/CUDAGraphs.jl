#=
    Macro definitions for @graphbreak, @scaptured, and @unsafe_scaptured.
=#

"""
    @graphbreak function f(args...) ... end

Annotate a function definition as a graph break. When `f` is called inside
`@scaptured` or `@unsafe_scaptured`, it automatically triggers a graph segment
boundary: the preceding kernels are captured as one graph segment, the function
body runs eagerly, and capture resumes for the next segment.

Outside of `@scaptured`/`@unsafe_scaptured`, the function runs normally with
negligible overhead (one branch on a global ref).

Supports all function definition forms:
- `@graphbreak function f(x, y) ... end`
- `@graphbreak function f(x::T) where T ... end`
- `@graphbreak function M.f(x) ... end` (qualified names)
- `@graphbreak f(x) = ...` (short form)
"""
macro graphbreak(fdef)
    return esc(_wrap_graphbreak(fdef))
end

function _wrap_graphbreak(ex::Expr)
    if ex.head === :function
        _wrap_function_form(ex)
    elseif ex.head === :(=) && ex.args[1] isa Expr && ex.args[1].head === :call
        # Short form: f(x) = body
        _wrap_short_form(ex)
    else
        error("@graphbreak: expected a function definition, got $(ex.head)")
    end
end

function _wrap_function_form(ex::Expr)
    sig = ex.args[1]
    body = ex.args[2]
    # Handle `where` clauses: function f(x::T) where T ... end
    # The signature is Expr(:where, call, T...) — we keep it as-is
    wrapped_body = _make_wrapped_body(body)
    return Expr(:function, sig, wrapped_body)
end

function _wrap_short_form(ex::Expr)
    sig = ex.args[1]
    body = ex.args[2]
    wrapped_body = _make_wrapped_body(body)
    return Expr(:(=), sig, wrapped_body)
end

function _make_wrapped_body(body)
    quote
        CUDAGraphs._at_break!() do
            $body
        end
    end
end

"""
    @scaptured begin ... end

Capture a block of GPU operations as segmented CUDA graphs, automatically splitting
at `@graphbreak` function calls. Re-captures every iteration and uses
`cuGraphExecUpdate` to cheaply detect topology changes.

Each textual occurrence of `@scaptured` gets its own cache (allocated as a
module-level constant).
"""
macro scaptured(body)
    cache_sym = gensym("scaptured_cache")
    # Allocate a module-level const cache per call site (same pattern as CUDA.@captured)
    @eval __module__ const $cache_sym = CUDAGraphs.SegmentedGraphCache()
    quote
        CUDAGraphs._run_scaptured!($(esc(cache_sym))) do
            $(esc(body))
        end
    end
end

"""
    @unsafe_scaptured cache begin ... end

Fast segmented graph capture with manual invalidation. Captures on first call,
replays from cache on subsequent calls. Call `invalidate!(cache)` when break
function arguments (especially value-type scalars) change.

`cache` must be a `SegmentedGraphCache` created by the user.
"""
macro unsafe_scaptured(cache, body)
    quote
        CUDAGraphs._run_unsafe_scaptured!($(esc(cache))) do
            $(esc(body))
        end
    end
end
