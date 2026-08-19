import JACC
JACC.@init_backend

using Gaugefields
using LinearAlgebra
using Test

const MPI = Gaugefields.AbstractGaugefields_module.MPI
const gather_and_bcast_matrix =
    Gaugefields.AbstractGaugefields_module.gather_and_bcast_matrix

MPI.Initialized() || MPI.Init()

function _maximum_orthogonality_error(field)
    identity_matrix = Matrix{eltype(field)}(I, size(field, 1), size(field, 2))
    return maximum(CartesianIndices(size(field)[3:end])) do site
        matrix = @view field[:, :, Tuple(site)...]
        norm(transpose(matrix) * matrix - identity_matrix)
    end
end

@testset "MPILattice gauge-field element types" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    global_size_4d = (2 * nprocs, 2, 2, 2)
    process_grid_4d = (nprocs, 1, 1, 1)

    for T in (Float32, Float64, ComplexF32, ComplexF64)
        fields = Initialize_Gaugefields(
            7,
            1,
            global_size_4d...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid_4d,
            elementtype=T,
            verbose_level=0,
        )
        field = fields[1]
        copied = similar(field)
        gathered = gather_and_bcast_matrix(field.U)

        @test eltype(field) === T
        @test eltype(field.U.A) === T
        @test eltype(copied) === T
        @test field.singleprecision == (T === Float32 || T === ComplexF32)
        @test gathered[:, :, 1, 1, 1, 1] == Matrix{T}(I, 7, 7)
        if T <: Real
            temps = [similar(field), similar(field)]
            expected_plaquette = 6 * prod(global_size_4d) * 7
            @test calculate_Plaquette(fields, temps) ≈ expected_plaquette
        end
    end

    legacy_default = Initialize_Gaugefields(
        2, 1, global_size_4d...;
        isMPILattice=true,
        PEs=process_grid_4d,
        verbose_level=0,
    )
    legacy_single = Initialize_Gaugefields(
        2, 1, global_size_4d...;
        isMPILattice=true,
        PEs=process_grid_4d,
        singleprecision=true,
        verbose_level=0,
    )
    explicit_type_wins = Initialize_Gaugefields(
        2, 1, global_size_4d...;
        isMPILattice=true,
        PEs=process_grid_4d,
        singleprecision=true,
        elementtype=Float64,
        verbose_level=0,
    )

    @test eltype(legacy_default[1]) === ComplexF64
    @test eltype(legacy_single[1]) === ComplexF32
    @test eltype(explicit_type_wins[1]) === Float64
    @test !explicit_type_wins[1].singleprecision

    for T in (Float32, Float64)
        fields = Initialize_Gaugefields(
            7,
            1,
            global_size_4d...;
            condition="hot",
            isMPILattice=true,
            PEs=process_grid_4d,
            elementtype=T,
            seed=0x1234,
            verbose_level=0,
        )
        tolerance = T === Float32 ? 3f-4 : 5e-12
        @test all(fields) do field
            gathered = gather_and_bcast_matrix(field.U)
            _maximum_orthogonality_error(gathered) < tolerance
        end
    end

    global_size_2d = (2 * nprocs, 2)
    process_grid_2d = (nprocs, 1)
    for T in (Float32, Float64)
        fields = Initialize_Gaugefields(
            7,
            1,
            global_size_2d...;
            condition="cold",
            isMPILattice=true,
            PEs=process_grid_2d,
            elementtype=T,
            verbose_level=0,
        )
        @test eltype(fields[1]) === T
        @test eltype(similar(fields[1])) === T
    end

    @test_throws ArgumentError Initialize_Gaugefields(
        2,
        1,
        global_size_4d...;
        isMPILattice=true,
        PEs=process_grid_4d,
        elementtype=Int64,
        verbose_level=0,
    )
    @test_throws ArgumentError Initialize_Gaugefields(
        2,
        0,
        global_size_4d...;
        elementtype=Float64,
        verbose_level=0,
    )
end
