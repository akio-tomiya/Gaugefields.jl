module HeatbathSU2BackendTests
using JACC
JACC.@init_backend
using Gaugefields, LatticeMatrices, StaticArrays, LinearAlgebra, Test
const HB=Gaugefields.heatbath_module
const distributed=get(ENV,"CHECK_MPI","0")=="1"
if distributed
    using MPI
    MPI.Initialized() || MPI.Init()
end
const rank=distributed ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
const nr=distributed ? MPI.Comm_size(MPI.COMM_WORLD) : 1
function probe(i,out,offset)
    global_i=i+offset
    rng=site_rng(RNGStreamKey(20261114),UInt64(global_i),Philox4x32())
    a=0.3+0.001global_i; b=0.2; c=-0.1; d=0.4
    V=MMatrix{2,2,ComplexF64}(a+im*d,-c+im*b,c+im*b,a-im*d)
    if global_i == 13
        V .= ComplexF64(NaN)
    elseif global_i == 14
        V .= ComplexF64(Inf)
    elseif global_i == 15
        V .= 0; V[1, 1] = 1
    elseif global_i == 16
        V .= 0
    end
    U=MMatrix{2,2,ComplexF64}(1,0,0,1)
    t1=MMatrix{2,2,ComplexF64}(undef); t2=MMatrix{2,2,ComplexF64}(undef)
    r=HB._su2_update_kp_core!(rng,U,V,1.9,2,(t1,t2),100000)
    out[1,i]=r[2]; out[2,i]=r[3]
    for j in 1:4
        out[2j+1,i]=real(U[j]); out[2j+2,i]=imag(U[j])
    end
end
count=div(16,nr); out=JACC.zeros(Float64,10,count)
JACC.parallel_for(count,probe,out,rank*count)
host=Array(out)
reference=zeros(10,count)
for i in 1:count; probe(i,reference,rank*count); end
@test isapprox(host,reference;rtol=2e-13,atol=2e-13)
for i in 1:count
    global_i = i + rank * count
    U = reshape(complex.(host[3:2:9, i], host[4:2:10, i]), 2, 2)
    if global_i > 12
        @test host[1, i] == 0 && host[2, i] == 0
        @test U == I
    else
        @test host[1, i] == 1 && host[2, i] >= 1
        @test U' * U ≈ I atol=2e-15
        @test det(U) ≈ 1 atol=2e-15
    end
end
result=distributed ? MPI.Gather(host,MPI.COMM_WORLD;root=0) : host
if rank==0
    result=reshape(result,10,16)
    file=get(ENV,"RESULT_FILE",nothing)
    if !isnothing(file)
        open(file,"w") do io
            for row in eachrow(result)
                println(io, join(row, " "))
            end
        end
    end
    println("HEATBATH_BACKEND nr=$nr max_reference_error=$(maximum(abs.(host-reference))) zero_staple_tries=$(result[2,end]) result=$file")
end

end
