module NERSCIOTests
using Gaugefields, LatticeMatrices, Test, LinearAlgebra, JACC
JACC.@init_backend
const comm=LatticeMatrices._default_communicator()
const rank=LatticeMatrices._comm_rank(comm)
const ranks=LatticeMatrices._comm_size(comm)
const snapshots=Dict{String,Vector{Float64}}()
pack(x)=Float64.(vcat(vec(real.(x)),vec(imag.(x))))
root(f)=Gaugefields._portable_jld2_root_operation(f,comm)
function changed_header(src,dst,key,value)
    open(src,"r") do io
        lines=String[]
        while true
            line=readline(io);push!(lines,startswith(line,key*" =") ? "$key = $value" : line)
            line=="END_HEADER" && break
        end
        payload=read(io)
        open(dst,"w") do out
            write(out,join(lines,"\n")*"\n");write(out,payload)
        end
    end
end
@testset "NERSC checked I/O on backend fields" begin
    directory=root(()->mktempdir())
    directory=Gaugefields.broadcast(directory,0,comm)
    try
        dims=(2,2,2,4);grid=(1,1,1,ranks)
        U=gauge_configuration(dims;colors=3,halo=1,start=:hot,seed=UInt64(1212),process_grid=grid,comm,verbose=0)
        originals=gather_and_bcast_matrix.(getproperty.(U,:U))
        file=joinpath(directory,"test.nersc")
        for precision in (32,64), endian in (:big,:little), compressed in (false,true)
            save_configuration(file,U;format=:nersc,precision,endian,compressed)
            for T in (ComplexF32,ComplexF64)
                V=load_configuration(file;format=:nersc,eltype=T,process_grid=grid,comm)
                @test eltype(V[1].U.A)==T
                tol=(precision==32 || T==ComplexF32) ? 3e-6 : 3e-12
                for mu in 1:4
                    actual=gather_and_bcast_matrix(V[mu].U)
                    @test actual ≈ originals[mu] rtol=tol atol=tol
                    @test !LatticeMatrices.halo_is_dirty(V[mu].U)
                    snapshots["$precision-$endian-$compressed-$T-$mu"]=pack(actual)
                end
                @test measure_plaquette(V) ≈ measure_plaquette(U) rtol=tol atol=tol
            end
            # Writers differ on the byte order of unsuffixed IEEE labels.
            alias=joinpath(directory,"alias.nersc")
            root(()->changed_header(file,alias,"FLOATING_POINT","IEEE$precision"))
            V=load_configuration(alias;format=:nersc,process_grid=grid,comm)
            @test measure_plaquette(V) ≈ measure_plaquette(U) atol=3e-6
        end
        # A Float32 field may be saved in the default Float64 file format.
        # Compression must allow its source rounding error in the third row.
        single=load_configuration(file;format=:nersc,eltype=ComplexF32,process_grid=grid,comm)
        save_configuration(file,single;format=:nersc,compressed=true)
        restored=load_configuration(file;format=:nersc,process_grid=grid,comm)
        @test eltype(restored[1].U.A)==ComplexF64
        for mu in 1:4
            @test gather_and_bcast_matrix(restored[mu].U) ≈ gather_and_bcast_matrix(single[mu].U) atol=2e-6 rtol=2e-6
        end
        before=gather_and_bcast_matrix.(getproperty.(U,:U))
        for (key,value) in (("CHECKSUM","00000000"),("PLAQUETTE","2.0"),("LINK_TRACE","2.0"),("DATATYPE","unsupported"),("FLOATING_POINT","IEEE16BIG"),("DIMENSION_1","0"),("DIMENSION_1","9223372036854775807"))
            bad=joinpath(directory,"bad.nersc")
            root(()->changed_header(file,bad,key,value))
            @test_throws ErrorException load_configuration!(U,bad;format=:nersc)
            @test all(gather_and_bcast_matrix(U[mu].U)==before[mu] for mu in 1:4)
        end
        bad=joinpath(directory,"truncated.nersc")
        root() do
            bytes=read(file);write(bad,bytes[1:end-1])
        end
        @test_throws ErrorException load_configuration!(U,bad;format=:nersc)
        @test_throws ErrorException load_configuration(bad;format=:nersc,process_grid=grid,comm)
        @test_throws ErrorException save_configuration(file,U;format=:nersc,precision=16)
        if !Sys.iswindows()
            # Cross-format read of the same links, including distributed GPU targets.
            ildg=joinpath(directory,"same.ildg")
            save_configuration(ildg,U;format=:ildg)
            V=copy_configuration(U)
            load_configuration!(V,ildg;format=:ildg)
            for mu in 1:4
                @test gather_and_bcast_matrix(V[mu].U) ≈ originals[mu] atol=1e-12
            end
        end
        if haskey(ENV,"G2_NERSC_REFERENCE")
            W=load_configuration(ENV["G2_NERSC_REFERENCE"];format=:nersc,eltype=ComplexF32,process_grid=grid,comm)
            @test gauge_lattice_size(W)==(8,8,8,4)
            @test measure_plaquette(W) ≈ 0.5038664469 atol=5e-6
            for mu in 1:4
                snapshots["external-$mu"]=pack(gather_and_bcast_matrix(W[mu].U))
            end
        end
    finally
        root(()->rm(directory;recursive=true,force=true))
    end
end
end
