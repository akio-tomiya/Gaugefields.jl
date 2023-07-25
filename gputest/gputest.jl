using CUDA
using Gaugefields

function gpu_print(U,i)
    a = U.U[i]
    @cuprintln("i = $i $(real(a))+$(imag(a))im")
    
    return 
end

function main()
    NC = 3
    NX = 4
    NY = 4
    NZ =4
    NT = 4
    U = Gaugefields.Gaugefields_4D_gpu(NC,NX,NY,NZ,NT)
    println(isbitstype(typeof(U)))
    U1 = Gaugefields.identityGaugefields_4D_gpu(NC, NX, NY, NZ, NT)
    U2 = Gaugefields.randomGaugefields_4D_gpu(NC, NX, NY, NZ, NT)
    #display(U.U)
    #display(U.U1)
    @cuda gpu_print(U,1)
    for i=1:3
        for j=1:3
            #println("$i $j")
            @cuda gpu_print(U1,(i-1)*3+j)
        end
    end
    substitute_U!(U,U1)
    for i=1:3
        for j=1:3
            #println("$i $j")
            @cuda gpu_print(U,(i-1)*3+j)
        end
    end
    for i=1:3
        for j=1:3
            #println("$i $j")
            @cuda gpu_print(U2,(i-1)*3+j)
        end
    end
    
end
main()