struct Adjoint_rep_Gaugefields_4D_wing{NC} <: Adjoint_rep_Gaugefields_4D{NC} 
    U::Array{ComplexF64,6}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NV::Int64
    NC::Int64
    mpi::Bool
    verbose_print::Verbose_print
end