function PrealocatedTwoformfields(B::Bfield{T,Dim}; num=10) where {T,Dim}
    twoformfield = B
    labeltype = Wilsonline{Dim}
    data = PreallocatedArray(B.u[1, 2]; num, labeltype, haslabel=true)
    TL = Union{Nothing,labeltype}

    tempgaugefields = PreallocatedArray(B.u[1, 2]; num=5)
    return PrealocatedTwoformfields{typeof(twoformfield),T,TL}(twoformfield, data, tempgaugefields)
end

function add_Wilsonline!(p::PrealocatedTwoformfields{TT,TG,TL}, w::Wilsonline{Dim}) where {TT<:Bfield,TG,TL,Dim}
    B = p.twoformfield
    t = p.data
    tempgaugefields = p.tempgaugefields
    uout, i = new_block_withlabel(t, w)
    temps, its_temps = get_block(tempgaugefields, 4)

    evaluate_Bplaquettes!(uout, w, B, temps)
    unused!(tempgaugefields, its_temps)
end

function load_Wilsonline(p::PrealocatedTwoformfields{TT,TG,TL}, w::Wilsonline{Dim}) where {TT<:Bfield,TG,TL,Dim}
    ti, index = load_block_withlabel(p.data, w)
    return ti, index
end


