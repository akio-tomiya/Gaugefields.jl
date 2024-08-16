struct STOUT_dataset{Dim}
    closedloop::Vector{Wilsonline{Dim}}
    Cμ::Vector{Vector{Wilsonline{Dim}}}
    dCμdUν::Matrix{Vector{DwDU{Dim}}} #(mu,nu) num loops
    dCμdagdUν::Matrix{Vector{DwDU{Dim}}} #(mu,nu) num loops
    #dCmudUnu::Dict{Tuple{Int64,Int64,Int64},Array{DwDU{Dim},1}}
end


function STOUT_dataset(closedloops; Dim=4)
    #Ci = #Dict{Tuple{Int64,Int64},Array{Wilsonline{Dim},1}}[]
    #dCmudUnu = #Dict{Tuple{Int64,Int64,Int64},Array{DwDU{Dim},1}}[]
    num = length(closedloops) #number of loops 

    Cμs = Vector{Vector{Wilsonline{Dim}}}(undef, Dim)
    for μ = 1:Dim
        Cμs[μ] = Vector{Wilsonline{Dim}}[] #set of staples. In the case of plaq, there are six staples. 
    end



    for i = 1:num
        glinks = closedloops[i]
        for μ = 1:Dim
            Cμ = make_Cμ(glinks, μ)
            for j = 1:length(Cμ)
                push!(Cμs[μ], Cμ[j])
            end
        end
    end

    CmudUnu = Matrix{Vector{DwDU{Dim}}}(undef, Dim, Dim)
    CmudagdUnu = Matrix{Vector{DwDU{Dim}}}(undef, Dim, Dim)

    #=
    CmudUnu = Array{Array{Dict{Tuple{Int8,Int8},Vector{DwDU{Dim}}},1},1}(undef,Dim)
    CmudUnudag = Array{Array{Dict{Tuple{Int8,Int8},Vector{DwDU{Dim}}},1},1}(undef,Dim)
    for μ=1:Dim
        Cμ = Cμs[μ]
        numCμ = length(Cμ)
        CmudUnu[μ] = Array{Dict{Int8,Vector{DwDU{Dim}}},1}(undef,numCμ )
        CmudUnudag[μ] = Array{Dict{Int8,Vector{DwDU{Dim}}},1}(undef,numCμ )


        for i=1:numCμ 
            Cμi = Cμ[i]
            CmudUnu[μ][i]=  Dict{Int8,Vector{DwDU{Dim}}}()
            CmudUnudag[μ][i]=  Dict{Int8,Vector{DwDU{Dim}}}()
            for ν=1:4
                CmudUnu[μ][i][ν] = derive_U(Cμi,ν)
                CmudUnudag[μ][i][ν] = derive_Udag(Cμi,ν)
            end
        end
    end

    return STOUT_dataset{Dim}(closedloops,Cμs,CmudUnu,CmudUnudag)
    =#


    for ν = 1:Dim
        for μ = 1:Dim
            CmudUnu[μ, ν] = Vector{DwDU{Dim}}[]
            CmudagdUnu[μ, ν] = Vector{DwDU{Dim}}[]
            Cμ = Cμs[μ]
            numCμ = length(Cμ)
            for j = 1:numCμ
                Cμj = Cμ[j]
                dCμjν = derive_U(Cμj, ν)
                numdCμjν = length(dCμjν)
                for k = 1:numdCμjν
                    push!(CmudUnu[μ, ν], dCμjν[k])
                end

                dCμjνdag = derive_U(Cμj', ν)
                numdCμjνdag = length(dCμjνdag)
                for k = 1:numdCμjνdag
                    push!(CmudagdUnu[μ, ν], dCμjνdag[k])
                end

            end
            #println("dC$(μ)/dU$(ν): ")
            #show(CmudUnu[μ,ν])
        end
    end


    #=
    for μ=1:Dim
        println("μ = $μ")
        show(Cμs[μ])
        for ν=1:Dim
            println("dC$(μ)/dU$(ν): ")
            show(CmudUnu[μ,ν])
        end

        for ν=1:Dim
            println("dCdag$(μ)/dU$(ν): ")
            show(CmudagdUnu[μ,ν])
        end
    end
    =#


    return STOUT_dataset{Dim}(closedloops, Cμs, CmudUnu, CmudagdUnu)
end

