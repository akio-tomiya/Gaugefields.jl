using Gaugefields

function main()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Nwing = 1
    Dim = 4

    U = Initialize_Gaugefields(NC, Nwing, NX, NY, NZ, NT, condition="cold")
    filename = "hoge.ildg"
    save_binarydata(U, filename; tempfile1="temp1.dat", tempfile2="temp2.dat")


    filename = "hoge.ildg"
    ildg = ILDG(filename)
    i = 1
    L = [NX, NY, NZ, NT]
    load_gaugefield!(U, i, ildg, L, NC)

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    comb = 6
    factor = 1 / (comb * U[1].NV * U[1].NC)
    @time plaq_t = calculate_Plaquette(U, temp1, temp2) * factor
    println("plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U, temp1, temp2)
    println("polyakov loop = $(real(poly)) $(imag(poly))")
end
main()