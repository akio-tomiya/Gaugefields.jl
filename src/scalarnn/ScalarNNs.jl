module ScalarNN_module
    import ..Abstractsmearing_module:CovNeuralnet
    import ..AbstractGaugefields_module:AbstractGaugefields,evaluate_gaugelinks!,add_U!,clear_U!,set_wing_U!
    
    import Wilsonloop:Wilsonline,make_staple

    struct ScalarNN_dataset{Dim}
        β::Float64
        closedloops::Vector{Wilsonline{Dim}}
        staples::Vector{Vector{Wilsonline{Dim}}}
    end

    struct ScalarNN{Dim,T}
        hascovnet::Bool
        covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
        dataset::Vector{ScalarNN_dataset{Dim}}
        _temp_U::Vector{T}
    end

    function ScalarNN_dataset(β,closedloops::Vector{Wilsonline{Dim}}) where Dim
        allstaples = Array{Vector{Wilsonline{Dim}},1}(undef,Dim)
        numloops = length(closedloops)
        for μ=1:Dim
            staples = Wilsonline{Dim}[]
            for i=1:numloops
                staples_i = make_staple(closedloops[i],μ)
                for j=1:length(staples_i)
                    push!(staples,staples_i[j])
                end
            end
            allstaples[μ] = staples
        end
        return ScalarNN_dataset{Dim}(β,closedloops,allstaples)
    end

    function calc_dSdUμ(dSdUμ,snet,μ,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        dSdUμ = snet._temp_U[4]
        calc_dSdUμ!(dSdUμ,snet,μ,U)
        return dSdUμ
    end

    function calc_dSdUμ!(dSdUμ,snet,μ,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        temp1 = snet._temp_U[1]
        temp2 = snet._temp_U[2]
        temp3 = snet._temp_U[3]
        numterm = length(snet.dataset)

        clear_U!(dSdUμ)
        for i=1:numterm
            dataset = snet.dataset[i]
            staples_μ =  dataset.staples[μ]
            evaluate_gaugelinks!(temp3,w,U,[temp1,temp2])
            add_U!(dSdUμ,β,temp3)
        end
        set_wing_U!(dSdUμ)
    end

    function apply_snet(snet::ScalarNN,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        uout = snet._temp_U[4]
        clear_U!(uout)

        apply_snet!(uout,snet,U)
        
        return uout
    end

    function apply_snet!(uout,snet::ScalarNN,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC,T}
        numterm = length(snet.dataset)
        temp1 = snet._temp_U[1]
        temp2 = snet._temp_U[2]
        temp3 = snet._temp_U[3]
        clear_U!(uout)

        for i=1:numterm
            dataset = snet.dataset[i]
            β = dataset.β
            w = dataset.closedloops
            evaluate_gaugelinks!(temp3,w,U,[temp1,temp2])
            add_U!(uout,β,temp3)
        end
        set_wing_U!(uout)

        return
    end

    function ScalarNN(U::Vector{<: AbstractGaugefields{NC,Dim}};hascovnet = false) where {NC,Dim}
        if hascovnet
            covneuralnet = CovNeuralnet(Dim=Dim)
        else
            covneuralnet = nothing
        end
        dataset = ScalarNN_dataset{Dim}[]
        num = 4
        _temp_U = Array{eltype(U)}(undef,num)
        for i=1:num
            _temp_U[i] = similar(U[1])
        end

        return ScalarNN{Dim,eltype(U)}(hascovnet,covneuralnet,dataset,_temp_U)
    end

    function Base.push!(snet::ScalarNN{Dim,T1},β::T,closedloops::Vector{Wilsonline{Dim}}) where {Dim,T <: Real,T1}
        dataset = ScalarNN_dataset(β,closedloops)
        push!(snet.dataset,dataset)
    end

    function Base.show(s::ScalarNN{Dim,T}) where {Dim,T}
        println("----------------------------------------------")
        println("Structure of scalar neural networks")
        println("num. of terms: ", length(s.dataset))
        for i=1:length(s.dataset)
            if i==1
                string = "st"
            elseif i==2
                string = "nd"
            elseif i==3
                string = "rd"
            else
                string = "th"
            end
            println("-------------------------------")
            println("      $i-$string term: ")
            println("          coefficient: ",s.dataset[i].β)
            println("      -------------------------")
            show(s.dataset[i].closedloops)
            println("      -------------------------")
        end
        println("----------------------------------------------")
    end


end