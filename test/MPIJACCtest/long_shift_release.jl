using Test
using MPI
using Gaugefields

const AGM = Gaugefields.AbstractGaugefields_module

Base.@noinline function abandon_long_gauge_shift(U, shift)
    shifted = shift_U(U, shift)
    return WeakRef(getfield(getfield(shifted, :U), :lease))
end

@testset "2D MPILattice long-shift ownership" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1)
    U = AGM.identityGaugefields_2D_MPILattice(
        1, 4 * nprocs, 4;
        NDW=1,
        PEs=process_grid,
        boundarycondition=ones(2),
        verbose_level=0,
    )
    pool = U.U.temps

    shifted = shift_U(U, (2, 0))
    @test isopen(shifted)
    @test count(pool._flagusing) == 1
    close(shifted)
    @test !isopen(shifted)
    @test count(pool._flagusing) == 0
    warmed_pool_size = length(pool)

    abandoned_lease = abandon_long_gauge_shift(U, (-2, 0))
    @test count(pool._flagusing) == 1
    GC.gc(true)
    @test abandoned_lease.value === nothing
    @test count(pool._flagusing) == 0
    @test length(pool) == warmed_pool_size
end

@testset "MPILattice long-shift ownership" begin
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    process_grid = (nprocs, 1, 1, 1)
    global_size = (4 * nprocs, 4, 4, 4)
    field() = AGM.identityGaugefields_4D_MPILattice(
        1, global_size...; NDW=1, PEs=process_grid, verbose_level=0)

    U = field()
    pool = U.U.temps
    initial_pool_size = length(pool)

    shifted = shift_U(U, (2, 0, 0, 0))
    @test isopen(shifted)
    @test count(pool._flagusing) == 1
    release!(shifted)
    release!(shifted)
    @test !isopen(shifted)
    @test count(pool._flagusing) == 0
    warmed_pool_size = length(pool)

    shifted = shift_U(U, (-2, 0, 0, 0))
    shifted_adjoint = shifted'
    release!(shifted_adjoint)
    @test !isopen(shifted)
    @test count(pool._flagusing) == 0

    abandoned_lease = abandon_long_gauge_shift(U, (2, 0, 0, 0))
    @test count(pool._flagusing) == 1
    GC.gc(true)
    @test abandoned_lease.value === nothing
    @test count(pool._flagusing) == 0
    @test initial_pool_size <= length(pool) == warmed_pool_size

    links = [field() for _ in 1:4]
    temp1 = similar(links[1])
    temp2 = similar(links[1])
    calculate_Polyakov_loop(links, temp1, temp2)
    @test all(count(link.U.temps._flagusing) == 0 for link in links)
    link_pool_sizes = map(link -> length(link.U.temps), links)
    for _ in 1:3
        calculate_Polyakov_loop(links, temp1, temp2)
        @test all(count(link.U.temps._flagusing) == 0 for link in links)
        @test map(link -> length(link.U.temps), links) == link_pool_sizes
    end
end
