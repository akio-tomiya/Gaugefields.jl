struct Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis} <:
       Adjoint_rep_Gaugefields_4D{NC,NumofBasis}
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
    NumofBasis::Int64
    generators::Union{Nothing,Generator}


    function Adjoint_rep_Gaugefields_4D_wing(
        NumofBasis,
        NC::T,
        NDW::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T;
        verbose_level = 2,
    ) where {T<:Integer}

        NV = NX * NY * NZ * NT
        U = zeros(
            ComplexF64,
            NumofBasis,
            NumofBasis,
            NX + 2NDW,
            NY + 2NDW,
            NZ + 2NDW,
            NT + 2NDW,
        )
        mpi = false
        verbose_print = Verbose_print(verbose_level)
        generators = Generator(NC)
        #=
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end
        =#

        return new{NC,NumofBasis}(
            U,
            NX,
            NY,
            NZ,
            NT,
            NDW,
            NV,
            NC,
            mpi,
            verbose_print,
            NumofBasis,
            generators,
        )
    end

end

function construct_Adjoint_rep_Gaugefields_4D_wing(
    Uin::Gaugefields_4D{NC};
    verbose_level = 2,
) where {NC}
    return Adjoint_rep_Gaugefields_4D_wing(Uin, verbose_level = verbose_level)
end

function Adjoint_rep_Gaugefields_4D_wing(
    Uin::Gaugefields_4D{NC};
    verbose_level = 2,
) where {NC}
    NX = Uin.NX
    NY = Uin.NY
    NZ = Uin.NZ
    NT = Uin.NT
    NumofBasis = ifelse(NC == 1, 1, NC^2 - 1)
    NDW = Uin.NDW
    Uadj = Adjoint_rep_Gaugefields_4D_wing(
        NumofBasis,
        NC,
        NDW,
        NX,
        NY,
        NZ,
        NT,
        verbose_level = verbose_level,
    )

    make_adjoint_rep!(Uadj, Uin)

    return Uadj
end

function Base.setindex!(x::Adjoint_rep_Gaugefields_4D_wing, v, i1, i2, i3, i4, i5, i6)
    @inbounds x.U[i1, i2, i3+x.NDW, i4+x.NDW, i5+x.NDW, i6+x.NDW] = v
end

@inline function Base.getindex(x::Adjoint_rep_Gaugefields_4D_wing, i1, i2, i3, i4, i5, i6)
    @inbounds return x.U[i1, i2, i3.+x.NDW, i4.+x.NDW, i5.+x.NDW, i6.+x.NDW]
end

struct Shifted_Adjoint_rep_Gaugefields_4D{NC,NumofBasis} <: Shifted_Gaugefields{NC,4}
    parent::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis}
    #parent::T
    shift::NTuple{4,Int8}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    outside::Bool

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_Adjoint_rep_Gaugefields_4D(
        U::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
        shift,
    ) where {NC,NumofBasis}
        outside = check_outside(U.NDW, shift)
        return new{NC,NumofBasis}(U, shift, U.NX, U.NY, U.NZ, U.NT, U.NDW, outside)
    end
end

function shift_U(U::Adjoint_rep_Gaugefields_4D_wing, ν::T) where {T<:Integer}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return Shifted_Adjoint_rep_Gaugefields_4D(U, shift)
end

function shift_U(
    U::TU,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TU<:Adjoint_rep_Gaugefields_4D}
    return Shifted_Adjoint_rep_Gaugefields_4D(U, shift)
end

function Base.setindex!(
    U::Shifted_Adjoint_rep_Gaugefields_4D,
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) 
    error("type $(typeof(U)) has no setindex method. This type is read only.")
end

@inline function Base.getindex(
    U::Shifted_Adjoint_rep_Gaugefields_4D{NC,NumofBasis},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC,NumofBasis}
    if U.outside != true
        #println("inside shift: ",U.shift,"\t",U.outside)
        @inbounds return U.parent[
            i1,
            i2,
            i3.+U.shift[1],
            i4.+U.shift[2],
            i5.+U.shift[3],
            i6.+U.shift[4],
        ]
    else
        #println("shift: ",U.shift,"\t",U.outside)

        i3_new = i3 + U.shift[1]
        i3_new += ifelse(i3_new > U.NX + U.NDW, -U.NX, 0)
        i3_new += ifelse(i3_new < 1 - U.NDW, U.NX, 0)
        i4_new = i4 + U.shift[2]
        i4_new += ifelse(i4_new > U.NY + U.NDW, -U.NY, 0)
        i4_new += ifelse(i4_new < 1 - U.NDW, U.NY, 0)
        i5_new = i5 + U.shift[3]
        i5_new += ifelse(i5_new > U.NZ + U.NDW, -U.NZ, 0)
        i5_new += ifelse(i5_new < 1 - U.NDW, U.NZ, 0)
        i6_new = i6 + U.shift[4]
        i6_new += ifelse(i6_new > U.NT + U.NDW, -U.NT, 0)
        i6_new += ifelse(i6_new < 1 - U.NDW, U.NT, 0)

        @inbounds return U.parent[i1, i2, i3_new, i4_new, i5_new, i6_new]
    end
    #function Base.getindex(U::Shifted_Gaugefields{T,4},i1,i2,i3,i4,i5,i6) where T <: Gaugefields_4D_wing
end


function Base.similar(U::T) where {T<:Adjoint_rep_Gaugefields_4D_wing}
    Uout = Adjoint_rep_Gaugefields_4D_wing(
        U.NumofBasis,
        U.NC,
        U.NDW,
        U.NX,
        U.NY,
        U.NZ,
        U.NT,
        verbose_level = U.verbose_print.level,
    )
    #identityGaugefields_4D_wing(U.NC,U.NX,U.NY,U.NZ,U.NT,U.NDW)
    return Uout
end

function Base.similar(U::Array{T,1}) where {T<:Adjoint_rep_Gaugefields_4D_wing}
    Uout = Array{T,1}(undef, 4)
    for μ = 1:4
        Uout[μ] = similar(U[μ])
    end
    return Uout
end


"""
Uadj = (1/2)*tr(Ta*U*Tb*Udag)
"""
function make_adjoint_rep!(
    Uadj::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    Uin,
) where {NC,NumofBasis}
    generators = Uadj.generators
    Ulocal = zeros(ComplexF64, NC, NC)
    NT = Uadj.NT
    NZ = Uadj.NZ
    NY = Uadj.NY
    NX = Uadj.NX


    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for k2 = 1:NC
                        for k1 = 1:NC
                            Ulocal[k1, k2] = Uin[k1, k2, ix, iy, iz, it]
                        end
                    end

                    for b = 1:NumofBasis
                        Tb = generators[b]
                        for a = 1:NumofBasis
                            Ta = generators[a]
                            Uadj[a, b, ix, iy, iz, it] = tr(Ta * Ulocal * Tb * Ulocal') / 2
                        end
                    end

                end
            end
        end
    end

end


function substitute_U!(
    a::Vector{T1},
    b::Vector{T2},
) where {T1<:Adjoint_rep_Gaugefields_4D_wing,T2<:Abstractfields}
    #function substitute_U!(a::Array{<: Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},1},b::Array{<: Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},1}) where {NC,NumofBasis}
    for i = 1:4
        substitute_U!(a[i], b[i])
    end
end

function substitute_U!(
    a::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    b::T2,
) where {NC,NumofBasis,T2<:Abstractfields}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for k2 = 1:NumofBasis
                        for k1 = 1:NumofBasis
                            @inbounds a[k1, k2, ix, iy, iz, it] = b[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(a)

end

function LinearAlgebra.mul!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
    b::T2,
) where {NC,NumofBasis,T1<:Number,T2<:Abstractfields}
    @inbounds for i = 1:length(c)
        c.U[i] = a * b.U[i]
    end
    return
end

function LinearAlgebra.mul!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
    b::T2,
    α::Ta,
    β::Tb,
) where {NC,NumofBasis,T1<:Abstractfields,T2<:Abstractfields,Ta<:Number,Tb<:Number}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for k2 = 1:NumofBasis
                        for k1 = 1:NumofBasis
                            c[k1, k2, ix, iy, iz, it] = β * c[k1, k2, ix, iy, iz, it]
                            @simd for k3 = 1:NumofBasis
                                c[k1, k2, ix, iy, iz, it] +=
                                    α *
                                    a[k1, k3, ix, iy, iz, it] *
                                    b[k3, k2, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function LinearAlgebra.mul!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
    b::T2,
    iseven::Bool,
) where {NC,T1<:Abstractfields,T2<:Abstractfields,NumofBasis}
    @assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for k2 = 1:NumofBasis
                            for k1 = 1:NumofBasis
                                c[k1, k2, ix, iy, iz, it] = 0
                                @simd for k3 = 1:NumofBasis
                                    c[k1, k2, ix, iy, iz, it] +=
                                        a[k1, k3, ix, iy, iz, it] *
                                        b[k3, k2, ix, iy, iz, it]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end



function LinearAlgebra.tr(
    a::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    b::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
) where {NC,NumofBasis}
    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT

    s = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for k = 1:NumofBasis
                        for k2 = 1:NumofBasis
                            s += a[k, k2, ix, iy, iz, it] * b[k2, k, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function LinearAlgebra.tr(
    a::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
) where {NC,NumofBasis}
    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT

    s = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        s += a[k, k, ix, iy, iz, it]
                        #println(a[k,k,ix,iy,iz,it])
                    end
                end
            end
        end
    end
    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function partial_tr(
    a::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    μ,
) where {NC,NumofBasis}
    NX = a.NX
    NY = a.NY
    NZ = a.NZ
    NT = a.NT

    if μ == 1
        s = 0
        ix = 1
        for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    #for ix=1:NX
                    @simd for k = 1:NumofBasis
                        s += a[k, k, ix, iy, iz, it]
                        #println(a[k,k,ix,iy,iz,it])
                    end

                    #end
                end
            end
        end
    elseif μ == 2
        s = 0
        iy = 1
        for it = 1:NT
            for iz = 1:NZ
                #for iy=1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        s += a[k, k, ix, iy, iz, it]
                        #println(a[k,k,ix,iy,iz,it])
                    end
                end
                #end
            end
        end
    elseif μ == 3
        s = 0
        iz = 1
        for it = 1:NT
            #for iz=1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        s += a[k, k, ix, iy, iz, it]
                        #println(a[k,k,ix,iy,iz,it])
                    end
                end
            end
            #end
        end
    else
        s = 0
        it = 1
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    @simd for k = 1:NumofBasis
                        s += a[k, k, ix, iy, iz, it]
                        #println(a[k,k,ix,iy,iz,it])
                    end
                end
            end
        end

    end



    #println(3*NT*NZ*NY*NX*NC)
    return s
end

function normalize_U!(
    u::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
) where {NC,NumofBasis}
    NX = u.NX
    NY = u.NY
    NZ = u.NZ
    NT = u.NT

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                @simd for ix = 1:NX
                    A = u[:, :, ix, iy, iz, it]
                    gramschmidt!(A)
                    u[:, :, ix, iy, iz, it] = A[:, :]
                end
            end
        end
    end

end



function add_U!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
) where {NC,T1<:Abstractfields,NumofBasis}
    @inbounds for i = 1:length(c.U)
        c.U[i] += a.U[i]
    end
    return
end

function add_U!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
    iseven::Bool,
) where {NC,T1<:Abstractfields,NumofBasis}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for k2 = 1:NumofBasis
                            @simd for k1 = 1:NumofBasis
                                c[k1, k2, ix, iy, iz, it] += a[k1, k2, ix, iy, iz, it]
                            end
                        end
                    end
                end
            end
        end
    end
end


function add_U!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    α::N,
    a::T1,
) where {NC,T1<:Abstractfields,N<:Number,NumofBasis}
    #@inbounds for i=1:length(c.U)
    #    c.U[i] += α*a.U[i]
    #end
    #return 

    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for k2 = 1:NumofBasis
                        @simd for k1 = 1:NumofBasis
                            c[k1, k2, ix, iy, iz, it] += α * a[k1, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end




function set_wing_U!(
    u::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
) where {NC,NumofBasis}
    NT = u.NT
    NY = u.NY
    NZ = u.NZ
    NX = u.NX
    NDW = u.NDW

    #X direction 
    #Now we send data

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for id = 1:NDW
                    for k2 = 1:NumofBasis
                        @simd for k1 = 1:NumofBasis
                            @inbounds u[k1, k2, -NDW+id, iy, iz, it] =
                                u[k1, k2, NX+(id-NDW), iy, iz, it]
                        end
                    end
                end
            end
        end
    end

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for id = 1:NDW
                    for k2 = 1:NumofBasis
                        @simd for k1 = 1:NumofBasis
                            @inbounds u[k1, k2, NX+id, iy, iz, it] =
                                u[k1, k2, id, iy, iz, it]
                        end
                    end
                end
            end
        end
    end


    #Y direction 
    #Now we send data
    for it = 1:NT
        for iz = 1:NZ
            for ix = -NDW+1:NX+NDW
                for id = 1:NDW
                    for k1 = 1:NumofBasis
                        @simd for k2 = 1:NumofBasis
                            @inbounds u[k1, k2, ix, -NDW+id, iz, it] =
                                u[k1, k2, ix, NY+(id-NDW), iz, it]
                        end
                    end
                end
            end
        end
    end

    for it = 1:NT
        for iz = 1:NZ
            for ix = -NDW+1:NX+NDW
                for id = 1:NDW
                    for k1 = 1:NumofBasis
                        @simd for k2 = 1:NumofBasis
                            @inbounds u[k1, k2, ix, NY+id, iz, it] =
                                u[k1, k2, ix, id, iz, it]
                        end
                    end
                end
            end
        end
    end

    #Z direction 
    #Now we send data
    for id = 1:NDW
        for it = 1:NT
            for iy = -NDW+1:NY+NDW
                for ix = -NDW+1:NX+NDW
                    for k1 = 1:NumofBasis
                        @simd for k2 = 1:NumofBasis
                            @inbounds u[k1, k2, ix, iy, id-NDW, it] =
                                u[k1, k2, ix, iy, NZ+(id-NDW), it]
                            @inbounds u[k1, k2, ix, iy, NZ+id, it] =
                                u[k1, k2, ix, iy, id, it]
                        end
                    end
                end
            end
        end
    end


    for id = 1:NDW
        for iz = -NDW+1:NZ+NDW
            for iy = -NDW+1:NY+NDW
                for ix = -NDW+1:NX+NDW
                    for k1 = 1:NumofBasis
                        @simd for k2 = 1:NumofBasis
                            @inbounds u[k1, k2, ix, iy, iz, id-NDW] =
                                u[k1, k2, ix, iy, iz, NT+(id-NDW)]
                            @inbounds u[k1, k2, ix, iy, iz, NT+id] =
                                u[k1, k2, ix, iy, iz, id]
                        end
                    end
                end
            end
        end
    end

    #display(u.g)
    #exit()

    return
end

function mul_skiplastindex!(
    c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},
    a::T1,
    b::T2,
) where {NC,NumofBasis,T1<:Abstractfields,T2<:Abstractfields}
    #@assert NC != 2 && NC != 3 "This function is for NC != 2,3"
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    #for it=1:NT
    it = 1
    for iz = 1:NZ
        for iy = 1:NY
            for ix = 1:NX
                for k2 = 1:NumofBasis
                    for k1 = 1:NumofBasis
                        c[k1, k2, ix, iy, iz, it] = 0

                        @simd for k3 = 1:NumofBasis
                            c[k1, k2, ix, iy, iz, it] +=
                                a[k1, k3, ix, iy, iz, it] * b[k3, k2, ix, iy, iz, it]
                        end
                    end
                end
            end
        end
    end
    #end
end
