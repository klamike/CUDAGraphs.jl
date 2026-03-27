#=
    Capture context — tracks the current mode and active cache.

    Modes:
    - :off          — normal execution, @graphbreak is a no-op wrapper
    - :capturing    — inside @scaptured or @unsafe_scaptured capture phase
    - :replaying    — inside @unsafe_scaptured replay phase (not used by @scaptured)
    - :recapturing  — inside @scaptured re-capture phase (update existing execs)
=#

mutable struct CaptureContext
    mode::Symbol
    cache::SegmentedGraphCache
    segment::Int
    stream::CUDA.CuStream
    capture_active::Bool
end

# Module-level context. For thread safety in v2, use task_local_storage.
const _CTX = Ref{CaptureContext}()

function _init_context!()
    _CTX[] = CaptureContext(:off, SegmentedGraphCache(), 0, CUDA.default_stream(), false)
end

@inline _mode() = _CTX[].mode
