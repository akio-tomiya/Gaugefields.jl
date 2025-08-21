using MPILattice
using Test
using MPI
import JACC
using LinearAlgebra
using InteractiveUtils
JACC.@init_backend
using MPI, JACC, StaticArrays



using Random

function normalizetest()
    MPI.Init()

    Random.seed!(1234)

    for NC = 2:4
        @testset "NC = $NC" begin
            normalizetest(NC)
        end
    end
end

function normalizetest(NC)
    dim = 4
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    gsize = (NX, NY, NZ, NT)
    #gsize = (NX, NY)
    nw = 1


    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = (n1, nprocs ÷ n1, 1, 1)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
    comm = M1.cart

    A = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
    M2 = LatticeMatrix(A, dim, PEs; nw)

    M5 = similar(M2)

    #shift = (3, 1, -1, -2)
    shift = (0, 0, 0, -2)
    M4 = Shifted_Lattice(M2, shift)
    substitute!(M5, M4)
    ix, iy, iz, it = 1, 1, 1, 1
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    ixp = ifelse(ixp < 1, ixp + NX, ixp)
    iyp = ifelse(iyp < 1, iyp + NY, iyp)
    izp = ifelse(izp < 1, izp + NZ, izp)
    itp = ifelse(itp < 1, itp + NT, itp)
    ixp = ifelse(ixp > NX, ixp - NX, ixp)
    iyp = ifelse(iyp > NY, iyp - NY, iyp)
    izp = ifelse(izp > NZ, izp - NZ, izp)
    itp = ifelse(itp > NT, itp - NT, itp)


    normalize_matrix!(M2)
    display(M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it] * M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it]')

    randomize_matrix!(M2)
    display(M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it])
    normalize_matrix!(M2)
    display(M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it] * M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it]')
    makeidentity_matrix!(M2)
    display(M2.A[:, :, nw+ix, nw+iy, nw+iz, nw+it])

    s = partial_trace(M2, 1)
    println(s)
end


function exptest(NC)
    dim = 4
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    gsize = (NX, NY, NZ, NT)
    #gsize = (NX, NY)
    nw = 1


    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    if length(ARGS) == 0
        n1 = nprocs ÷ 2
        if n1 == 0
            n1 = 1
        end
        PEs = (n1, nprocs ÷ n1, 1, 1)
    else
        PEs = Tuple(parse.(Int64, ARGS))
    end
    M1 = LatticeMatrix(NC, NC, dim, gsize, PEs; nw)
    comm = M1.cart

    A = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
    M2 = LatticeMatrix(A, dim, PEs; nw)

    M5 = similar(M2)

    #shift = (3, 1, -1, -2)
    shift = (0, 0, 0, -2)
    M4 = Shifted_Lattice(M2, shift)
    substitute!(M5, M4)
    ix, iy, iz, it = 1, 1, 1, 1
    ixp = ix + shift[1]
    iyp = iy + shift[2]
    izp = iz + shift[3]
    itp = it + shift[4]
    ixp = ifelse(ixp < 1, ixp + NX, ixp)
    iyp = ifelse(iyp < 1, iyp + NY, iyp)
    izp = ifelse(izp < 1, izp + NZ, izp)
    itp = ifelse(itp < 1, itp + NT, itp)
    ixp = ifelse(ixp > NX, ixp - NX, ixp)
    iyp = ifelse(iyp > NY, iyp - NY, iyp)
    izp = ifelse(izp > NZ, izp - NZ, izp)
    itp = ifelse(itp > NT, itp - NT, itp)

    display(A[:, :, ixp, iyp, izp, itp])
    display(M5.A[:, :, nw+ix, nw+iy, nw+iz, nw+it])

    A5 = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
    M5 = LatticeMatrix(A5, dim, PEs; nw)

    dA5n = zero(A5)


    dM5 = similar(M5)
    println(tr(M5))

    return


    A3 = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
    M3 = LatticeMatrix(A3, dim, PEs; nw)

    A1 = zeros(ComplexF64, NC, NC, NX, NY, NZ, NT)

    println("AdagB")
    M2t = M2'
    mul!(M1, M2t, M3)
    display(M1.A[:, :, 2, 2, 2, 2])
    display(A[:, :, 1, 1, 1, 1]' * A3[:, :, 1, 1, 1, 1])


    M2t = M2'
    M3t = M3'
    println("ABdag")
    mul!(M1, M2, M3t)
    display(M1.A[:, :, 2, 2, 2, 2])
    display(A[:, :, 1, 1, 1, 1] * A3[:, :, 1, 1, 1, 1]')

    println("AdagBdag")
    mul!(M1, M2t, M3t)
    display(M1.A[:, :, 2, 2, 2, 2])
    display(A[:, :, 1, 1, 1, 1]' * A3[:, :, 1, 1, 1, 1]')

    #mul!(A1, A', A3)
    #println(sum(A1))

    println("substitute test")
    @testset "substitute test" begin
        display(M2.A[:, :, 3, 2, 2, 2])
        substitute!(M1, M2)
        display(M1.A[:, :, 3, 2, 2, 2])
        @test M2.A ≈ M1.A atol = 1e-6

        shift = (1, 0, 0, 0)
        M3 = Shifted_Lattice(M2, shift)
        substitute!(M1, M3)
        display(M1.A[:, :, 2, 2, 2, 2])
    end

    println("mul test")
    @testset "mul test" begin
        A2 = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
        M2 = LatticeMatrix(A2, dim, PEs; nw)

        A3 = rand(ComplexF64, NC, NC, NX, NY, NZ, NT)
        M3 = LatticeMatrix(A3, dim, PEs; nw)

        mul!(M1, M2', M3)
        #println("MPILattice")
        a = M1.A[:, :, 2, 2, 2, 2]
        #display(a)
        #display(M1.A[:, :, 2, 2, 2, 2])
        #println("original")
        ao = A2[:, :, 1, 1, 1, 1]' * A3[:, :, 1, 1, 1, 1]
        #display(ao)
        @test a ≈ ao atol = 1e-6
        #display(A2[:, :, 1, 1, 1, 1]' * A3[:, :, 1, 1, 1, 1])
        shift = (1, 0, 0, 0)
        M4 = Shifted_Lattice(M3, shift)
        mul!(M1, M2', M4')
        #println("MPILattice")
        a = M1.A[:, :, 2, 2, 2, 2]
        #display(M1.A[:, :, 2, 2, 2, 2])
        #display(a)
        #println("original")
        ao = A2[:, :, 1, 1, 1, 1]' * A3[:, :, 2, 1, 1, 1]'
        @test a ≈ ao atol = 1e-6
        #display(ao)
        #display(A2[:, :, 1, 1, 1, 1]' * A3[:, :, 2, 1, 1, 1]')
    end

    return



    t = 0.3
    expt!(M1, M2, t)
    display(M1.A[:, :, 2, 2, 2, 2])
    a = exp(t * A[:, :, 1, 1, 1, 1])
    display(a)

    A = rand(NC, NC, NX, NY, NZ, NT)
    v = deepcopy(A)
    ix, iy, iz, it = 1, 1, 1, 1
    fac1N = 1 / NC
    tri = 0.0
    for k = 1:NC
        tri += imag(v[k, k, ix, iy, iz, it])
    end
    tri *= fac1N
    for k = 1:NC
        v[k, k, ix, iy, iz, it] =
            (imag(v[k, k, ix, iy, iz, it]) - tri) * im
    end


    for k1 = 1:NC
        for k2 = k1+1:NC
            vv =
                0.5 * (
                    v[k1, k2, ix, iy, iz, it] -
                    conj(v[k2, k1, ix, iy, iz, it])
                )
            v[k1, k2, ix, iy, iz, it] = vv
            v[k2, k1, ix, iy, iz, it] = -conj(vv)
        end
    end

    println("Traceless antihermitian matrix:")

    M2 = LatticeMatrix(v, dim, PEs; nw)
    expt!(M1, M2, t)
    display(M1.A[:, :, 2, 2, 2, 2])
    a = exp(t * v[:, :, 1, 1, 1, 1])
    display(a)

    sm = TALattice(M2)
    expt!(M1, sm, t)
    display(M1.A[:, :, 2, 2, 2, 2])


end
function latticetest4D()
    MPI.Init()

    Random.seed!(1234)
    a0 = rand(3, 3)
    a = Mat3{Complex{Float64}}(a0[:]...)
    t = 0.4
    b = exp3x3_pade(a, t)

    b0 = Matrix(b)
    display(b0)
    display(exp(t * a0))

    B = expm_pade13(a0, t)
    display(B)

    @testset "Matrix exponential" begin
        for NC = 2:4
            @testset "NC = $NC" begin
                for i = 1:10
                    a0 = rand(NC, NC)
                    t = rand()
                    #println("i = $i, NC = $NC, t = $t")
                    #@time expt(a0, t)
                    #@time exp(t * a0)
                    c1 = expt(a0, t)
                    c2 = exp(t * a0)
                    @test c1 ≈ c2 atol = 1e-6
                    #display(expt(a0, t))
                    #display(exp(t * a0))
                end
                a0 = rand(NC, NC)
                t = rand()
                println("NC = $NC")
                println("original")
                display(expt(a0, t))
                println("Base")
                display(exp(t * a0))
            end
        end
    end

    NC = 3
    for NC = 2:4
        exptest(NC)
    end


    return

    println("SM2")
    SM2 = TALattice(M2)
    return

    A3 = rand(NC, NC, NX, NY, NZ, NT)
    M3 = LatticeMatrix(A3, dim, PEs; nw)
    mul!(M1, M2, M3)
    println(allsum(M1))
    println(allsum(M2))
    println(sum(A))
    #display(M1)

    #return
    shift = (1, 0, 0, 0)
    M3 = Shifted_Lattice(M2, shift)
    return

    if M2.myrank == 0
        display(A[:, :, NX, NY, NZ, NT])
        display(M2.A[:, :, 1, 1, 1, 1])
    end


    if M2.myrank == 0
        display(A[:, :, NX, 1, 2, 1])
        display(M2.A[:, :, 1, 1+nw, 2+nw, 1+nw])
    end
end


@testset "MPILattice.jl" begin


    # Write your tests here.

    #latticetest4D()

    normalizetest()


end
