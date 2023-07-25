using CUDA
using Gaugefields

function gpu_print(U,i)
    a = U.U[i]
    @cuprintln("$(real(a))+$(imag(a)) im)")
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
    #display(U.U)
    @cuda gpu_print(U,1)
end
main()