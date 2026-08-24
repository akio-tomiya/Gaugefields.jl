module Communication

import LatticeMatrices: SerialCommunicator

const SERIAL_COMMUNICATOR = SerialCommunicator()

"""Resolve the default communicator once, during field construction."""
function default_communicator()
    extension = Base.get_extension(parentmodule(@__MODULE__), :GaugefieldsMPIExt)
    return extension === nothing ? SERIAL_COMMUNICATOR :
        extension.default_communicator()
end

@inline resolve_communicator(comm) = comm
@inline resolve_communicator(::Nothing) = default_communicator()

@inline communicator_ready(::SerialCommunicator) = true
@inline prepare_communicator(comm::SerialCommunicator) = comm
@inline comm_size(::SerialCommunicator) = 1
@inline comm_rank(::SerialCommunicator) = 0
@inline barrier(::SerialCommunicator) = nothing

@inline function broadcast(value, root::Integer, ::SerialCommunicator)
    iszero(root) || throw(ArgumentError(
        "the serial communicator only contains root rank 0, got root $root"))
    return value
end

@inline function broadcast!(value, root::Integer, ::SerialCommunicator)
    iszero(root) || throw(ArgumentError(
        "the serial communicator only contains root rank 0, got root $root"))
    return value
end

@inline allreduce_sum(value, ::SerialCommunicator) = value

end
