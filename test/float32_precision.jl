module GaugePrecisionTests
using Gaugefields, LatticeMatrices, LinearAlgebra, Test, JACC
JACC.@init_backend
const comm=LatticeMatrices._default_communicator()
const ranks=LatticeMatrices._comm_size(comm)
const snapshots=Dict{String,Vector{Float64}}()
pack(x)=Float64.(vcat(vec(real.(x)),vec(imag.(x))))
@testset "Gaugefield Float32 precision and portable loading" begin
    dims=(2,2,2,4); grid=(1,1,1,ranks)
    for T in (ComplexF32,ComplexF64), nc in (2,3,4)
        U=gauge_configuration(dims;colors=nc,halo=1,start=:hot,seed=UInt64(911),eltype=T,process_grid=grid,comm,verbose=0)
        V=smear(U,stout_smearing(U;rho=0.1))
        @test isfinite(measure_plaquette(V))
        for mu in 1:4
            v=gather_and_bcast_matrix(V[mu].U)
            @test all(isfinite,v)
            for site in CartesianIndices(dims)
                m=v[:,:,Tuple(site)...]
                @test m'*m ≈ I atol=(T==ComplexF32 ? 2e-5 : 2e-12)
            end
            snapshots["stout-$T-$nc-$mu"]=pack(v)
        end
        if nc<=3
            native=smear(U,StoutSmearing(rho=0.1))
            @test isfinite(measure_plaquette(native))
            for mu in 1:4
                snapshots["native-$T-$nc-$mu"]=pack(gather_and_bcast_matrix(native[mu].U))
            end
        end
        # All ranks must use the same file, including GPU+MPI validation.
        filename=joinpath(tempdir(),"gauge-precision-$(get(ENV,"G2_TEST_ID","serial"))-$nc.jld2")
        original=gauge_configuration(dims;colors=nc,halo=1,start=:hot,seed=UInt64(915),eltype=ComplexF64,process_grid=grid,comm,verbose=0)
        save_configuration(filename,original)
        load_configuration!(U,filename)
        for mu in 1:4
            @test gather_and_bcast_matrix(U[mu].U) == T.(gather_and_bcast_matrix(original[mu].U))
        end
        LatticeMatrices._comm_rank(comm)==0 && rm(filename;force=true)
    end
end
end
