struct Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis} <: Adjoint_rep_Gaugefields_4D{NC} 
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


    function Adjoint_rep_Gaugefields_4D_wing(NC::T,NDW::T,NX::T,NY::T,NZ::T,NT::T;verbose_level = 2) where T<: Integer
        NumofBasis = ifelse(NC == 1,1,NC^2-1)
        NV = NX*NY*NZ*NT
        U = zeros(ComplexF64,NumofBasis,NumofBasis,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW)
        mpi = false
        verbose_print = Verbose_print(verbose_level )
        if NC <= 3
            generators = nothing
        else
            generators = Generator(NC)
        end

        return new{NC,NumofBasis}(U,NX,NY,NZ,NT,NDW,NV,NC,mpi,verbose_print,
                    NumofBasis,generators)
    end

end

function Adjoint_rep_Gaugefields_4D_wing(Uin::Gaugefields_4D{NC})  where NC
    NX = Uin.NX
    NY = Uin.NY
    NZ = Uin.NZ
    NT = Uin.NT
    NDW = Uin.NDW
    Uadj = Adjoint_rep_Gaugefields_4D_wing(NC,NDW,NX,NY,NZ,NT,verbose_level = U.verbose_level)

    make_adjoint_rep!(Uadj,Uin)

    return Uadj
end

function Base.setindex!(x::Adjoint_rep_Gaugefields_4D_wing,v,i1,i2,i3,i4,i5,i6) 
    @inbounds x.U[i1,i2,i3 + x.NDW,i4 + x.NDW,i5 + x.NDW,i6 + x.NDW] = v
end

@inline function Base.getindex(x::Adjoint_rep_Gaugefields_4D_wing,i1,i2,i3,i4,i5,i6) 
    @inbounds return x.U[i1,i2,i3 .+ x.NDW,i4 .+ x.NDW,i5 .+ x.NDW,i6 .+ x.NDW]
end

"""
Uadj = (1/2)*tr(Ta*U*Tb*Udag)
"""
function make_adjoint_rep!(Uadj::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},Uin) where {NC,NumofBasis}
    generators = Uadj.generators
    Ulocal = zeros(ComplexF64,NC,NC)
    NT = Uadj.NT
    NZ = Uadj.NZ
    NY = Uadj.NY
    NX = Uadj.NX


    @inbounds for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    for k2=1:NC                            
                        for k1=1:NC
                            Ulocal[k1,k2] = Uin[k1,k2,ix,iy,iz,it]
                        end
                    end

                    for b=1:NumofBasis
                        Tb = generators[b]
                        for a=1:NumofBasis
                            Ta = generators[a]
                            Uadj[a,b,ix,iy,iz,it] = tr(Ta*Ulocal*Tb*Ulocal')/2
                        end
                    end

                end
            end
        end
    end

end


function substitute_U!(a::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},b::T2) where {NC, NumofBasis,T2 <: Abstractfields}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    for k2=1:NumofBasis                            
                        for k1=1:NumofBasis
                            @inbounds a[k1,k2,ix,iy,iz,it] = b[k1,k2,ix,iy,iz,it]
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(a)

end

function LinearAlgebra.mul!(c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},a::T1,b::T2) where {NC,NumofBasis,T1 <: Number,T2 <: Abstractfields}
    @inbounds for i=1:length(c)
        c.U[i] = a*b.U[i]
    end
    return
end

function LinearAlgebra.mul!(c::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis},a::T1,b::T2,α::Ta,β::Tb) where {NC,NumofBasis,T1 <: Abstractfields,T2 <: Abstractfields,Ta <: Number, Tb <: Number}
    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    for k2=1:NumofBasis                            
                        for k1=1:NumofBasis
                            c[k1,k2,ix,iy,iz,it] = β*c[k1,k2,ix,iy,iz,it] 
                            @simd for k3=1:NumofBasis
                                c[k1,k2,ix,iy,iz,it] += α*a[k1,k3,ix,iy,iz,it]*b[k3,k2,ix,iy,iz,it] 
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(c)
end

function set_wing_U!(u::Adjoint_rep_Gaugefields_4D_wing{NC,NumofBasis}) where {NC,NumofBasis}
    NT = u.NT
    NY = u.NY
    NZ = u.NZ
    NX = u.NX
    NDW = u.NDW

    #X direction 
    #Now we send data

    for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for id=1:NDW
                    for k2=1:NumofBasis
                        @simd for k1=1:NumofBasis
                            @inbounds u[k1,k2,-NDW+id,iy,iz,it] = u[k1,k2,NX+(id-NDW),iy,iz,it]
                        end
                    end
                end
            end
        end
    end

    for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for id=1:NDW
                    for k2=1:NumofBasis
                        @simd for k1=1:NumofBasis
                            @inbounds u[k1,k2,NX+id,iy,iz,it] = u[k1,k2,id,iy,iz,it]
                        end
                    end
                end
            end
        end
    end


    #Y direction 
    #Now we send data
    for it=1:NT
        for iz=1:NZ
            for ix=-NDW+1:NX+NDW
                for id=1:NDW
                    for k1=1:NumofBasis
                        @simd for k2=1:NumofBasis
                            @inbounds u[k1,k2,ix,-NDW+id,iz,it] = u[k1,k2,ix,NY+(id-NDW),iz,it]
                        end
                    end
                end
            end
        end
    end

    for it=1:NT
        for iz=1:NZ
            for ix=-NDW+1:NX+NDW
                for id=1:NDW
                    for k1=1:NumofBasis
                        @simd for k2=1:NumofBasis
                            @inbounds u[k1,k2,ix,NY+id,iz,it] = u[k1,k2,ix,id,iz,it]
                        end
                    end
                end
            end
        end
    end

    #Z direction 
    #Now we send data
    for id=1:NDW
        for it=1:NT
            for iy=-NDW+1:NY+NDW
                for ix=-NDW+1:NX+NDW
                    for k1=1:NumofBasis
                        @simd for k2=1:NumofBasis
                            @inbounds u[k1,k2,ix,iy,id-NDW,it] = u[k1,k2,ix,iy,NZ+(id-NDW),it]
                            @inbounds u[k1,k2,ix,iy,NZ+id,it] = u[k1,k2,ix,iy,id,it]
                        end
                    end
                end
            end
        end
    end


    for id=1:NDW
        for iz=-NDW+1:NZ+NDW
            for iy=-NDW+1:NY+NDW
                for ix=-NDW+1:NX+NDW
                    for k1=1:NumofBasis
                        @simd for k2=1:NumofBasis
                            @inbounds u[k1,k2,ix,iy,iz,id-NDW] = u[k1,k2,ix,iy,iz,NT+(id-NDW)]
                            @inbounds u[k1,k2,ix,iy,iz,NT+id] = u[k1,k2,ix,iy,iz,id]
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
