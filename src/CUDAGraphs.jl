"""
    CUDAGraphs

Segmented CUDA graph capture with automatic graph breaks.

Provides two capture modes and one annotation macro:

- `@graphbreak`: Annotate a function definition whose body cannot be captured in a
  CUDA graph (e.g., CUDSS batch solves). Any call to this function — no matter how
  deep in the call stack — automatically triggers a graph segment boundary when
  executing inside `@scaptured` or `@unsafe_scaptured`.

- `@scaptured`: Safe, automatic mode. Re-captures every iteration and uses
  `cuGraphExecUpdate` to cheaply update cached executables when topology is unchanged.
  No closures, no manual invalidation needed.

- `@unsafe_scaptured cache body`: Fast, manual mode. Captures once and replays from
  cached graph segments + stored closures. User must call `invalidate!(cache)` when
  break function arguments change. Maximum performance for hot loops where invalidation
  points are known.

# Example

```julia
using CUDAGraphs

# Annotate non-graphable functions (once, at definition):
@graphbreak function my_library_call!(solver, rhs)
    LibrarySolver.solve!(solver, rhs)
end

# Safe mode (general purpose):
for i in 1:N
    @scaptured begin
        step!(solver)   # calls my_library_call! deep inside
    end
end

# Fast mode (when you know exactly when to invalidate):
cache = SegmentedGraphCache()
for i in 1:N
    config_changed && invalidate!(cache)
    @unsafe_scaptured cache begin
        step!(solver)
    end
end
```

!!! warning "Requirements"
    - Julia must be run with `--check-bounds=no`. Bounds checking on CuArray fancy
      indexing triggers allocations and stream syncs that are forbidden during capture.
    - All GPU memory must be pre-allocated. No allocations in the captured block.
    - Broadcasts must be fused (use `.=`, not standalone `.+` etc.) to avoid temporaries.
    - `@graphbreak` functions should be robust to garbage input data (during capture,
      preceding kernels are recorded but not executed, so buffer contents are stale).
"""
module CUDAGraphs

using CUDA

export @graphbreak, @scaptured, @unsafe_scaptured,
       SegmentedGraphCache, invalidate!, set_debug_capture_failures!, set_enabled!

const _ENABLED = Ref(true)
const _DEBUG_CAPTURE_FAILURES = Ref(false)

set_enabled!(enabled::Bool=true) = (_ENABLED[] = enabled)
set_debug_capture_failures!(enabled::Bool=true) = (_DEBUG_CAPTURE_FAILURES[] = enabled)

function _report_capture_failure(mode::Symbol, err, bt)
    _DEBUG_CAPTURE_FAILURES[] || return
    msg = sprint() do io
        print(io, "CUDAGraphs ", mode, " capture failed: ")
        showerror(io, err, bt)
    end
    @warn msg
    return
end

@inline _in_unsafe_capture() = _CTX[].mode === :capturing && _CTX[].capture_active
@inline _in_unsafe_replay() = _CTX[].mode === :replaying
@inline _in_unsafe_scaptured() = _in_unsafe_capture() || _in_unsafe_replay()

function _isvalid_ctx(ctx::CUDA.CuContext)
    if CUDA.driver_version() >= v"12"
        id_ref = Ref{CUDA.Culonglong}()
        res = CUDA.unchecked_cuCtxGetId(ctx, id_ref)
        res == CUDA.ERROR_CONTEXT_IS_DESTROYED && return false
        res != CUDA.SUCCESS && CUDA.throw_api_error(res)
        return ctx.id == id_ref[]
    else
        version_ref = Ref{CUDA.Cuint}()
        res = CUDA.unchecked_cuCtxGetApiVersion(ctx, version_ref)
        res == CUDA.ERROR_INVALID_CONTEXT && return false
        return true
    end
end

include("cache.jl")
include("context.jl")
include("raw_graph_api.jl")
include("break.jl")
include("scaptured.jl")
include("unsafe_scaptured.jl")
include("macros.jl")

function __init__()
    _init_context!()
    _ENABLED[] = !(get(ENV, "JULIA_CUDAGRAPHS_DISABLE", "") in ("1", "true", "TRUE", "yes", "YES"))
    _DEBUG_CAPTURE_FAILURES[] = get(ENV, "JULIA_CUDAGRAPHS_DEBUG_CAPTURE_FAILURES", "") in ("1", "true", "TRUE", "yes", "YES")
end

end # module
