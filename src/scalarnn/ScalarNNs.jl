module ScalarNN_module
    import ..Abstractsmearing_module:CovNeuralnet
    import Wilsonloop:Wilsonline

    struct ScalarNN_dataset{Dim}
        closedloops::Vector{Wilsonline{Dim}}
        staples::Vector{Vector{Wilsonline{Dim}}}
    end

    struct ScalarNN{Dim}
        βs::Vector{Float64}
        hascovnet::Bool
        covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
        dataset::Union{Nothing,Vector{ScalarNN_dataset{Dim}}}
    end



    function ScalarNN()

    end
end