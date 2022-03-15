module GaugeAction_module
    import ..Abstractsmearing_module:CovNeuralnet
    import ..AbstractGaugefields_module:AbstractGaugefields,evaluate_gaugelinks!,add_U!,clear_U!,set_wing_U!,getvalue,
                            evaluate_gaugelinks_eachsite! 
    
    import Wilsonloop:Wilsonline,make_staple
    using LinearAlgebra
    using InteractiveUtils

    struct GaugeAction_dataset{Dim}
        β::Float64
        closedloops::Vector{Wilsonline{Dim}}
        staples::Vector{Vector{Wilsonline{Dim}}}
    end

    struct GaugeAction{Dim,T}
        hascovnet::Bool
        covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
        dataset::Vector{GaugeAction_dataset{Dim}}
        _temp_U::Vector{T}
    end

    function GaugeAction_dataset(β,closedloops::Vector{Wilsonline{Dim}}) where Dim
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
        return GaugeAction_dataset{Dim}(β,closedloops,allstaples)
    end

    function get_temporary_gaugefields(S::GaugeAction)
        return S._temp_U
    end

    function calc_dSdUμ(S,μ,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        dSdUμ = similar(U[1])
        calc_dSdUμ!(dSdUμ,S,μ,U)
        return dSdUμ
    end

    function calc_dSdUμ!(dSdUμ,S,μ,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        temp1 = S._temp_U[1]
        temp2 = S._temp_U[2]
        temp3 = S._temp_U[3]
        numterm = length(S.dataset)

        clear_U!(dSdUμ)
        for i=1:numterm
            dataset = S.dataset[i]
            β =  dataset.β
            staples_μ =  dataset.staples[μ]
            evaluate_gaugelinks!(temp3,staples_μ,U,S._temp_U)
            
            #println("temp3 in dSdUμ! ",getvalue(temp3,1,1,1,1,1,1))
            add_U!(dSdUμ,β,temp3)
            #println("dSdUμ! ",getvalue(dSdUμ,1,1,1,1,1,1))
        end
        set_wing_U!(dSdUμ)

    end

    function evaluate_GaugeAction(S::GaugeAction,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        temp1 = S._temp_U[4]
        evaluate_GaugeAction_untraced!(temp1,S,U)
        value = tr(temp1)
        return value
    end

    function evaluate_GaugeAction_untraced(S::GaugeAction,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC}
        uout = similar(U[1])
        clear_U!(uout)

        evaluate_GaugeAction_untraced!(uout,S,U)
        
        return uout
    end

    function evaluate_staple_eachindex!(mat_U,μ,S::GaugeAction,U::Vector{<: AbstractGaugefields{NC,Dim}},mat_temps,indices...) where {Dim,NC,T}
        temp3 = mat_temps[5]
        numterm = length(S.dataset)
        mat_U .= 0
        for i=1:numterm
            dataset = S.dataset[i]
            β = dataset.β
            staples_μ =  dataset.staples[μ]
            evaluate_gaugelinks_eachsite!(temp3,staples_μ,U,view(mat_temps,1:4),indices...)
            mat_U .+= β*temp3
        end
    end

    function evaluate_GaugeAction_untraced!(uout,S::GaugeAction,U::Vector{<: AbstractGaugefields{NC,Dim}}) where {Dim,NC,T}
        numterm = length(S.dataset)
        temp1 = S._temp_U[1]
        temp2 = S._temp_U[2]
        temp3 = S._temp_U[3]
        clear_U!(uout)

        for i=1:numterm
            dataset = S.dataset[i]
            β = dataset.β
            w = dataset.closedloops
            evaluate_gaugelinks!(temp3,w,U,[temp1,temp2])
            add_U!(uout,β,temp3)
        end
        set_wing_U!(uout)

        return
    end

    function GaugeAction(U::Vector{<: AbstractGaugefields{NC,Dim}};hascovnet = false) where {NC,Dim}
        if hascovnet
            covneuralnet = CovNeuralnet(Dim=Dim)
        else
            covneuralnet = nothing
        end
        dataset = GaugeAction_dataset{Dim}[]
        num = 4
        _temp_U = Array{eltype(U)}(undef,num)
        for i=1:num
            _temp_U[i] = similar(U[1])
        end


        return GaugeAction{Dim,eltype(U)}(hascovnet,covneuralnet,dataset,_temp_U)
    end

    function Base.push!(S::GaugeAction{Dim,T1},β::T,closedloops::Vector{Wilsonline{Dim}}) where {Dim,T <: Real,T1}
        dataset = GaugeAction_dataset(β,closedloops)
        push!(S.dataset,dataset)
    end

    function Base.show(s::GaugeAction{Dim,T}) where {Dim,T}
        println("----------------------------------------------")
        println("Structure of the actions for Gaugefields")
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