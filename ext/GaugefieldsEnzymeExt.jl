module GaugefieldsEnzymeExt
using LinearAlgebra
using LatticeMatrices
using Enzyme
using JACC
import LatticeMatrices:
    Wiltinger_derivative!,
    toann,
    DiffArg,
    NoDiffArg,
    Enzyme_derivative!,
    enzyme_duplicated,
    fold_halo_dim_to_core_grad!,
    zero_halo_region!

using Gaugefields
#import LatticeMatrices: diff, nodiff, toann, Wiltinger_derivative!, Wiltinger!
#export diff, nodiff, Wiltinger_derivative!
import LatticeMatrices: Wiltinger!
import Gaugefields:
    Wiltinger_U!,
    diff,
    nodiff,
    Wiltinger_derivative!,
    enzyme_md_action,
    md_action_workspace,
    md_potential,
    md_force!
import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_MPILattice,
    _release_shifted_U!

const LCNN = Gaugefields.LCNN
const ER = Enzyme.EnzymeRules

@inline _lcnn_shadow(::Nothing) = nothing
@inline _lcnn_shadow(x::Base.RefValue) = _lcnn_shadow(x[])
@inline _lcnn_shadow(x) = x

@inline _lcnn_primal(x) = hasproperty(x, :val) ? x.val : x

# `tr(::LatticeMatrix)` is a JACC reduction.  Differentiating through the
# reduction implementation would make its CUDA kernel arguments active, which
# KernelAbstractions deliberately rejects.  Trace is instead an atomic LCNN
# boundary: its pullback adds the scalar output cotangent to every site's
# diagonal.  The identity field is passed from ModelWorkspace so this remains
# allocation-free and uses the ordinary backend-aware `add_U!` kernel on CPU,
# GPU, and MPI fields.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LCNN._trace_readout)},
    ::Type{RT},
    readout::ER.Const,
    channels::ER.Annotation,
    identity::ER.Annotation,
) where {RT}
    value = LCNN._trace_readout(readout.val, channels.val, identity.val)
    primal = ER.needs_primal(cfg) ? value : nothing
    shadow = ER.needs_shadow(cfg) ? zero(value) : nothing
    RetT = ER.augmented_rule_return_type(cfg, RT, shadow)
    return RetT(primal, shadow, shadow)
end

@inline function _lcnn_trace_cotangent(
    shadow,
    channel::Int,
    ::Val{:real},
)
    return shadow[channel]
end

@inline function _lcnn_trace_cotangent(
    shadow,
    channel::Int,
    ::Val{:imag},
)
    return im * shadow[channel]
end


@inline function _lcnn_trace_cotangent(
    shadow,
    channel::Int,
    ::Val{:both},
)
    return shadow[2 * channel - 1] + im * shadow[2 * channel]
end

@inline function _lcnn_trace_cotangent(
    shadow,
    channel::Int,
    ::Val{:complex},
)
    return shadow[channel]
end

function ER.reverse(
    ::ER.RevConfig,
    ::ER.Const{typeof(LCNN._trace_readout)},
    _dret,
    shadow,
    readout::ER.Const,
    channels::ER.Annotation,
    identity::ER.Annotation,
)
    channel_cotangents = hasproperty(channels, :dval) ?
        _lcnn_shadow(channels.dval) : nothing
    if shadow isa AbstractVector && channel_cotangents isa AbstractVector
        reduction = LCNN.trace_reduction(readout.val)
        scale = reduction === :mean ?
            inv(prod(size(first(channels.val))[3:end])) : 1
        component = Val(LCNN.trace_component(readout.val))
        for channel in eachindex(channel_cotangents)
            coefficient = scale * _lcnn_trace_cotangent(
                shadow, channel, component,
            )
            Gaugefields.add_U!(
                channel_cotangents[channel], coefficient, identity.val,
            )
        end
        fill!(shadow, zero(eltype(shadow)))
    end
    return (nothing, nothing, nothing)
end

# Stop Enzyme at the dynamic Wilson-line evaluator.  The matching reverse rule
# below implements the four analytic link contributions of every plaquette;
# when links are Const it simply consumes the feature cotangents.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LCNN._plaquette_features_only!)},
    ::Type{RT},
    channels::ER.Annotation,
    layer::ER.Const,
    U::ER.Annotation,
    workspace::ER.Annotation,
) where {RT}
    LCNN._plaquette_features_only!(
        channels.val, layer.val, U.val, workspace.val,
    )
    RetT = ER.augmented_rule_return_type(cfg, RT, nothing)
    return RetT(nothing, nothing, nothing)
end

# Hide halo/scratch-buffer bookkeeping from Enzyme.  The feature pullback is
# transport along the opposite oriented link, while `_lcnn_link_pullback!`
# supplies the two endpoint terms when the links themselves are active.  The
# rule is attached to the atomic operation, so arbitrary layer tuples remain
# ordinary differentiable Julia composition.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LCNN._parallel_transport_only!)},
    ::Type{RT},
    output::ER.Annotation,
    links::ER.Annotation,
    feature::ER.Annotation,
    direction::ER.Annotation,
    step::ER.Annotation,
    workspace::ER.Annotation,
) where {RT}
    LCNN._parallel_transport_only!(
        output.val,
        links.val,
        feature.val,
        _lcnn_primal(direction),
        _lcnn_primal(step),
        workspace.val,
    )
    RetT = ER.augmented_rule_return_type(cfg, RT, nothing)
    return RetT(nothing, nothing, nothing)
end

function _lcnn_link_pullback!(
    dlink,
    link,
    feature,
    output_cotangent,
    direction::Int,
    step::Int,
    dimension::Int,
    work::LCNN.ReferenceWorkspace,
)
    iszero(step) && return nothing
    abs(step) == 1 || throw(ArgumentError(
        "the LCNN link pullback currently supports radius-one transport",
    ))
    shift = LCNN._axis_shift(Val(dimension), direction, step)

    if step > 0
        shifted_feature = Gaugefields.shift_U(feature, shift)
        try
            # Y = U X_+ U'; dU = H U X_+' + H' U X_+.
            mul!(work.local_cotangent, output_cotangent, link)
            mul!(work.inverse_transport, work.local_cotangent, shifted_feature')
            Gaugefields.add_U!(dlink, work.inverse_transport)
            mul!(work.local_cotangent, output_cotangent', link)
            mul!(work.inverse_transport, work.local_cotangent, shifted_feature)
            Gaugefields.add_U!(dlink, work.inverse_transport)
        finally
            _release_shifted_U!(shifted_feature)
        end
    else
        forward_shift = LCNN._axis_shift(
            Val(dimension), direction, 1,
        )
        shifted_cotangent = Gaugefields.shift_U(output_cotangent, forward_shift)
        try
            # Y(x) = U(x-mu)' X(x-mu) U(x-mu).
            mul!(work.local_cotangent, feature', link)
            mul!(work.inverse_transport, work.local_cotangent, shifted_cotangent)
            Gaugefields.add_U!(dlink, work.inverse_transport)
            mul!(work.local_cotangent, feature, link)
            mul!(
                work.inverse_transport,
                work.local_cotangent,
                shifted_cotangent',
            )
            Gaugefields.add_U!(dlink, work.inverse_transport)
        finally
            _release_shifted_U!(shifted_cotangent)
        end
    end
    return nothing
end

function _lcnn_transport_prefix!(
    links,
    feature,
    direction::Int,
    orientation::Int,
    steps::Int,
    work::LCNN.ReferenceWorkspace,
)
    iszero(steps) && return feature
    source = feature
    for segment in 1:steps
        destination = work.path_temps[isodd(segment) ? 1 : 2]
        LCNN._parallel_transport_one_step_only!(
            destination, links, source, direction, orientation, work,
        )
        source = destination
    end
    return source
end

# Pull back T_s^n(W) by replaying the short Wilson line.  This keeps the
# analytic rule at the radius-one JACC/LatticeMatrices kernel boundary while
# allowing arbitrary kernel_size/dilation in an ordinary Julia layer stack.
function _lcnn_transport_pullback!(
    feature_cotangent,
    link_cotangents,
    links,
    feature,
    output_cotangent,
    direction::Int,
    step::Int,
    work::LCNN.ReferenceWorkspace,
)
    if iszero(step)
        feature_cotangent isa Gaugefields.AbstractGaugefields &&
            Gaugefields.add_U!(feature_cotangent, output_cotangent)
        return nothing
    end

    orientation = sign(step)
    distance = abs(step)
    cotangent = output_cotangent
    for segment in distance:-1:1
        prefix = _lcnn_transport_prefix!(
            links, feature, direction, orientation, segment - 1, work,
        )
        if link_cotangents isa AbstractVector
            _lcnn_link_pullback!(
                link_cotangents[direction],
                links[direction],
                prefix,
                cotangent,
                direction,
                orientation,
                length(links),
                work,
            )
        end
        next_cotangent = cotangent === work.transported_cotangent ?
            work.transport_input_cotangent : work.transported_cotangent
        LCNN._parallel_transport_one_step_only!(
            next_cotangent,
            links,
            cotangent,
            direction,
            -orientation,
            work,
        )
        cotangent = next_cotangent
    end
    feature_cotangent isa Gaugefields.AbstractGaugefields &&
        Gaugefields.add_U!(feature_cotangent, cotangent)
    return nothing
end

function ER.reverse(
    ::ER.RevConfig,
    ::ER.Const{typeof(LCNN._parallel_transport_only!)},
    _dret,
    _tape,
    output::ER.Annotation,
    links::ER.Annotation,
    feature::ER.Annotation,
    direction::ER.Annotation,
    step::ER.Annotation,
    workspace::ER.Annotation,
)
    doutput = hasproperty(output, :dval) ? _lcnn_shadow(output.dval) : nothing
    dfeature = hasproperty(feature, :dval) ? _lcnn_shadow(feature.dval) : nothing
    dlinks = hasproperty(links, :dval) ? _lcnn_shadow(links.dval) : nothing

    if doutput isa Gaugefields.AbstractGaugefields
        step_value = Int(_lcnn_primal(step))
        _lcnn_transport_pullback!(
            dfeature,
            dlinks,
            links.val,
            feature.val,
            doutput,
            Int(_lcnn_primal(direction)),
            step_value,
            workspace.val,
        )
        Gaugefields.clear_U!(doutput)
    end

    if hasproperty(workspace, :dval)
        dworkspace = _lcnn_shadow(workspace.dval)
        if dworkspace isa LCNN.ReferenceWorkspace
            Gaugefields.clear_U!(dworkspace.transport_product)
            Gaugefields.clear_U!(dworkspace.transported)
        end
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

@inline function _lcnn_augmented_descriptor(layer::LCNN.LCB, index::Int)
    return LCNN._local_descriptor(layer, index)
end

@inline _lcnn_field_data(field) = getproperty(field, :U)

function _lcnn_materialize_cotangent!(destination, source, adjoint_source::Bool)
    Gaugefields.clear_U!(destination)
    Gaugefields.add_U!(destination, adjoint_source ? source' : source)
    return destination
end

function _lcnn_accumulate_channel!(
    destination,
    cotangent,
    channel_kind::Symbol,
)
    channel_kind === :identity && return nothing
    Gaugefields.add_U!(
        destination,
        channel_kind === :adjoint ? cotangent' : cotangent,
    )
    return nothing
end

# The LCB rule is the layer/kernel boundary. The surrounding model tuple and
# loss remain ordinary Julia code, so Enzyme is free to compose any number of
# layers. Parameters, input features, and links may independently be active;
# link activity is handled by analytic one-step transport pullbacks, composed
# for longer Wilson lines.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LCNN._lcb_only!)},
    ::Type{RT},
    outputs::ER.Annotation,
    layer::ER.Const,
    links::ER.Annotation,
    channels::ER.Annotation,
    parameters::ER.Annotation,
    workspace::ER.Annotation,
) where {RT}
    LCNN._lcb_only!(
        outputs.val,
        layer.val,
        links.val,
        channels.val,
        parameters.val,
        workspace.val,
    )
    RetT = ER.augmented_rule_return_type(cfg, RT, nothing)
    return RetT(nothing, nothing, nothing)
end

function ER.reverse(
    ::ER.RevConfig,
    ::ER.Const{typeof(LCNN._lcb_only!)},
    _dret,
    _tape,
    outputs::ER.Annotation,
    layer_annotation::ER.Const,
    links::ER.Annotation,
    channels::ER.Annotation,
    parameters::ER.Annotation,
    workspace::ER.Annotation,
)
    output_cotangents = hasproperty(outputs, :dval) ?
        _lcnn_shadow(outputs.dval) : nothing
    channel_cotangents = hasproperty(channels, :dval) ?
        _lcnn_shadow(channels.dval) : nothing
    parameter_cotangent = hasproperty(parameters, :dval) ?
        _lcnn_shadow(parameters.dval) : nothing
    link_cotangents = hasproperty(links, :dval) ?
        _lcnn_shadow(links.dval) : nothing

    if !(output_cotangents isa AbstractVector) ||
       !(channel_cotangents isa AbstractVector)
        return (nothing, nothing, nothing, nothing, nothing, nothing)
    end

    layer = layer_annotation.val
    primal_links = links.val
    primal_channels = channels.val
    primal_parameters = parameters.val
    work = workspace.val
    weights = primal_parameters.weight
    dweights = parameter_cotangent === nothing ? nothing :
        parameter_cotangent.weight
    local_count = LCNN.augmented_channel_count(layer)
    transported_count = size(weights, 3)

    for transported_index in 1:transported_count
        transported_kind, transported_channel, displacement =
            LCNN._transported_descriptor(
                layer, Val(length(primal_links)), transported_index,
            )
        direction, step = displacement
        transported = LCNN._transported_channel!(
            layer,
            primal_links,
            primal_channels,
            transported_index,
            work,
        )
            raw_transported = if transported_kind === :adjoint
                iszero(step) ?
                    primal_channels[transported_channel] :
                    work.transported
            else
                transported
            end
            transported_adjoint = if transported_kind === :identity
                work.identity
            elseif transported_kind === :adjoint
                raw_transported
            else
                raw_transported'
            end

            for local_index in 1:local_count
                local_kind, local_channel =
                    _lcnn_augmented_descriptor(layer, local_index)
                local_feature = LCNN._augmented_channel(
                    layer,
                    primal_channels,
                    work.identity,
                    local_index,
                )
                local_adjoint = if local_kind === :identity
                    work.identity
                elseif local_kind === :adjoint
                    primal_channels[local_channel]
                else
                    local_feature'
                end

                mul!(work.bilinear_product, local_feature, transported)
                Gaugefields.clear_U!(work.product_cotangent)
                for output_index in eachindex(output_cotangents)
                    coefficient = weights[
                        output_index, local_index, transported_index,
                    ]
                    Gaugefields.add_U!(
                        work.product_cotangent,
                        coefficient,
                        output_cotangents[output_index],
                    )
                    if dweights !== nothing
                        dweights[
                            output_index, local_index, transported_index,
                        ] += real(dot(
                            _lcnn_field_data(output_cotangents[output_index]),
                            _lcnn_field_data(work.bilinear_product),
                        ))
                    end
                end

                if local_kind !== :identity
                    mul!(
                        work.local_cotangent,
                        work.product_cotangent,
                        transported_adjoint,
                    )
                    _lcnn_accumulate_channel!(
                        channel_cotangents[local_channel],
                        work.local_cotangent,
                        local_kind,
                    )
                end

                if transported_kind !== :identity
                    mul!(
                        work.transported_cotangent,
                        local_adjoint,
                        work.product_cotangent,
                    )
                    raw_transport_cotangent = _lcnn_materialize_cotangent!(
                        work.transport_input_cotangent,
                        work.transported_cotangent,
                        transported_kind === :adjoint,
                    )
                    _lcnn_transport_pullback!(
                        channel_cotangents[transported_channel],
                        link_cotangents,
                        primal_links,
                        primal_channels[transported_channel],
                        raw_transport_cotangent,
                        direction,
                        step,
                        work,
                    )
                end
            end
    end

    Gaugefields.clear_U!.(output_cotangents)
    if hasproperty(workspace, :dval)
        dworkspace = _lcnn_shadow(workspace.dval)
        if dworkspace isa LCNN.ReferenceWorkspace
            for field in (
                dworkspace.transport_product,
                dworkspace.transported,
                dworkspace.bilinear_product,
                dworkspace.product_cotangent,
                dworkspace.local_cotangent,
                dworkspace.transported_cotangent,
                dworkspace.transport_input_cotangent,
                dworkspace.inverse_transport,
            )
                Gaugefields.clear_U!(field)
            end
        end
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

function _lcnn_lcb_pullback!(
    channel_cotangents,
    link_cotangents,
    parameter_cotangent,
    output_cotangents,
    layer,
    primal_links,
    primal_channels,
    primal_parameters,
    work::LCNN.ReferenceWorkspace,
)
    weights = primal_parameters.weight
    dweights = parameter_cotangent === nothing ? nothing :
        parameter_cotangent.weight
    local_count = LCNN.augmented_channel_count(layer)
    transported_count = size(weights, 3)

    for transported_index in 1:transported_count
        transported_kind, transported_channel, displacement =
            LCNN._transported_descriptor(
                layer, Val(length(primal_links)), transported_index,
            )
        direction, step = displacement
        transported = LCNN._transported_channel!(
            layer,
            primal_links,
            primal_channels,
            transported_index,
            work,
        )
        raw_transported = if transported_kind === :adjoint
            iszero(step) ? primal_channels[transported_channel] :
            work.transported
        else
            transported
        end
        transported_adjoint = if transported_kind === :identity
            work.identity
        elseif transported_kind === :adjoint
            raw_transported
        else
            raw_transported'
        end

        for local_index in 1:local_count
            local_kind, local_channel =
                _lcnn_augmented_descriptor(layer, local_index)
            local_feature = LCNN._augmented_channel(
                layer,
                primal_channels,
                work.identity,
                local_index,
            )
            local_adjoint = if local_kind === :identity
                work.identity
            elseif local_kind === :adjoint
                primal_channels[local_channel]
            else
                local_feature'
            end

            mul!(work.bilinear_product, local_feature, transported)
            Gaugefields.clear_U!(work.product_cotangent)
            for output_index in eachindex(output_cotangents)
                coefficient = weights[
                    output_index, local_index, transported_index,
                ]
                Gaugefields.add_U!(
                    work.product_cotangent,
                    coefficient,
                    output_cotangents[output_index],
                )
                if dweights !== nothing
                    dweights[
                        output_index, local_index, transported_index,
                    ] += real(dot(
                        _lcnn_field_data(output_cotangents[output_index]),
                        _lcnn_field_data(work.bilinear_product),
                    ))
                end
            end

            if local_kind !== :identity
                mul!(
                    work.local_cotangent,
                    work.product_cotangent,
                    transported_adjoint,
                )
                _lcnn_accumulate_channel!(
                    channel_cotangents[local_channel],
                    work.local_cotangent,
                    local_kind,
                )
            end

            if transported_kind !== :identity
                mul!(
                    work.transported_cotangent,
                    local_adjoint,
                    work.product_cotangent,
                )
                raw_transport_cotangent = _lcnn_materialize_cotangent!(
                    work.transport_input_cotangent,
                    work.transported_cotangent,
                    transported_kind === :adjoint,
                )
                _lcnn_transport_pullback!(
                    channel_cotangents[transported_channel],
                    link_cotangents,
                    primal_links,
                    primal_channels[transported_channel],
                    raw_transport_cotangent,
                    direction,
                    step,
                    work,
                )
            end
        end
    end
    Gaugefields.clear_U!.(output_cotangents)
    return nothing
end

function _lcnn_plaquette_link_pullback!(
    link_cotangents,
    links,
    output_cotangent,
    mu::Int,
    nu::Int,
    work::LCNN.ReferenceWorkspace,
)
    dimension = length(links)
    shift_mu = LCNN._axis_shift(Val(dimension), mu, 1)
    shift_nu = LCNN._axis_shift(Val(dimension), nu, 1)
    inverse_mu = LCNN._axis_shift(Val(dimension), mu, -1)
    inverse_nu = LCNN._axis_shift(Val(dimension), nu, -1)
    A = links[mu]
    D = links[nu]
    B = Gaugefields.shift_U(D, shift_mu)
    C = Gaugefields.shift_U(A, shift_nu)
    try
        # P = A B C' D'.  Each block below forms the cotangent of
        # one occurrence before undoing its shift/adjoint wrapper.

        # dA = H (B C' D')'.
        mul!(work.transport_product, B, C')
        mul!(work.transported, work.transport_product, D')
        mul!(work.bilinear_product, output_cotangent, work.transported')
        Gaugefields.add_U!(link_cotangents[mu], work.bilinear_product)

        # dB = A' H (C' D')', followed by the inverse +mu shift.
        mul!(work.transport_product, C', D')
        mul!(work.transported, A', output_cotangent)
        mul!(work.bilinear_product, work.transported, work.transport_product')
        shifted_gradient = Gaugefields.shift_U(work.bilinear_product, inverse_mu)
        try
            Gaugefields.add_U!(link_cotangents[nu], shifted_gradient)
        finally
            _release_shifted_U!(shifted_gradient)
        end

        # Cotangent of C' is (A B)' H D; undo C(x)=U_mu(x+nu)
        # and then the adjoint operation.
        mul!(work.transport_product, A, B)
        mul!(work.transported, work.transport_product', output_cotangent)
        mul!(work.bilinear_product, work.transported, D)
        shifted_gradient = Gaugefields.shift_U(work.bilinear_product, inverse_nu)
        try
            Gaugefields.add_U!(link_cotangents[mu], shifted_gradient')
        finally
            _release_shifted_U!(shifted_gradient)
        end

        # Cotangent of D' is (A B C')' H; undo the adjoint.
        mul!(work.transport_product, A, B)
        mul!(work.transported, work.transport_product, C')
        mul!(work.bilinear_product, work.transported', output_cotangent)
        Gaugefields.add_U!(link_cotangents[nu], work.bilinear_product')
    finally
        _release_shifted_U!(C)
        _release_shifted_U!(B)
    end
    return nothing
end

function ER.reverse(
    ::ER.RevConfig,
    ::ER.Const{typeof(LCNN._plaquette_features_only!)},
    _dret,
    _tape,
    channels::ER.Annotation,
    _layer::ER.Const,
    U::ER.Annotation,
    workspace::ER.Annotation,
)
    shadow_channels = hasproperty(channels, :dval) ?
        _lcnn_shadow(channels.dval) : nothing
    shadow_links = hasproperty(U, :dval) ? _lcnn_shadow(U.dval) : nothing
    if shadow_channels isa AbstractVector && shadow_links isa AbstractVector
        channel = 1
        for mu in eachindex(U.val)
            for nu in (mu + 1):length(U.val)
                Gaugefields.set_wing_U!(shadow_channels[channel])
                _lcnn_plaquette_link_pullback!(
                    shadow_links,
                    U.val,
                    shadow_channels[channel],
                    mu,
                    nu,
                    workspace.val,
                )
                channel += 1
            end
        end
    end
    if hasproperty(channels, :dval)
        shadow_channels isa AbstractVector && Gaugefields.clear_U!.(shadow_channels)
    end
    if hasproperty(workspace, :dval)
        shadow_workspace = _lcnn_shadow(workspace.dval)
        if shadow_workspace isa LCNN.ReferenceWorkspace
            Gaugefields.clear_U!.(shadow_workspace.path_temps)
        end
    end
    return (nothing, nothing, nothing, nothing)
end

# LExp is the link-valued boundary of an LCNN. Its reverse pass is kept
# analytic: the matrix-exponential Fréchet pullback runs in LatticeMatrices'
# JACC kernel, while the surrounding flexible feature stack remains ordinary
# Enzyme composition.
function ER.augmented_primal(
    cfg::ER.RevConfig,
    ::ER.Const{typeof(LCNN._lexp_only!)},
    ::Type{RT},
    output::ER.Annotation,
    links::ER.Annotation,
    channels::ER.Annotation,
    weights::ER.Annotation,
    combinations::ER.Annotation,
    generators::ER.Annotation,
    exponentials::ER.Annotation,
    exponential_temps::ER.Annotation,
) where {RT}
    LCNN._lexp_only!(
        output.val,
        links.val,
        channels.val,
        weights.val,
        combinations.val,
        generators.val,
        exponentials.val,
        exponential_temps.val,
    )
    RetT = ER.augmented_rule_return_type(cfg, RT, nothing)
    return RetT(nothing, nothing, nothing)
end

function ER.reverse(
    ::ER.RevConfig,
    ::ER.Const{typeof(LCNN._lexp_only!)},
    _dret,
    _tape,
    output::ER.Annotation,
    links::ER.Annotation,
    channels::ER.Annotation,
    weights::ER.Annotation,
    combinations::ER.Annotation,
    generators::ER.Annotation,
    exponentials::ER.Annotation,
    exponential_temps::ER.Annotation,
)
    doutput = hasproperty(output, :dval) ?
        _lcnn_shadow(output.dval) : nothing
    dlinks = hasproperty(links, :dval) ?
        _lcnn_shadow(links.dval) : nothing
    dchannels = hasproperty(channels, :dval) ?
        _lcnn_shadow(channels.dval) : nothing
    dweights = hasproperty(weights, :dval) ?
        _lcnn_shadow(weights.dval) : nothing
    dcombinations = hasproperty(combinations, :dval) ?
        _lcnn_shadow(combinations.dval) : nothing
    dgenerators = hasproperty(generators, :dval) ?
        _lcnn_shadow(generators.dval) : nothing
    dexponentials = hasproperty(exponentials, :dval) ?
        _lcnn_shadow(exponentials.dval) : nothing

    if doutput isa AbstractVector
        all(scratch -> scratch isa AbstractVector,
            (dcombinations, dgenerators, dexponentials)) ||
            throw(ArgumentError(
                "the LExp Enzyme rule requires a duplicated LinkModelWorkspace",
            ))
        for direction in eachindex(output.val)
            output_cotangent = doutput[direction]

            if dlinks isa Union{Tuple,AbstractVector}
                # V = E*U gives dU = E' * dV.
                mul!(
                    dgenerators[direction],
                    exponentials.val[direction]',
                    output_cotangent,
                )
                Gaugefields.add_U!(
                    dlinks[direction], dgenerators[direction],
                )
            end

            # In the bilinear trace convention expected by
            # exp_ta_pullback!, the cotangent of E is U*dV'.
            mul!(
                dcombinations[direction],
                links.val[direction],
                output_cotangent',
            )
            LatticeMatrices.exp_ta_pullback!(
                _lcnn_field_data(dexponentials[direction]),
                _lcnn_field_data(dcombinations[direction]),
                _lcnn_field_data(combinations.val[direction]),
                1,
            )
            # Convert the bilinear-trace representative to the real
            # Frobenius cotangent: dA = -TA(R).
            Gaugefields.Traceless_antihermitian!(
                dgenerators[direction], dexponentials[direction],
            )

            for channel in eachindex(channels.val)
                if dchannels isa AbstractVector
                    Gaugefields.add_U!(
                        dchannels[channel],
                        -weights.val[direction, channel],
                        dgenerators[direction],
                    )
                end
                if dweights isa AbstractMatrix
                    dweights[direction, channel] -= real(dot(
                        _lcnn_field_data(dgenerators[direction]),
                        _lcnn_field_data(channels.val[channel]),
                    ))
                end
            end
            Gaugefields.clear_U!(output_cotangent)
        end
    end
    return (
        nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing,
    )
end

function _lcnn_lexp_pullback!(
    link_cotangents,
    channel_cotangents,
    weight_cotangent,
    output_cotangents,
    links,
    channels,
    weights,
    combinations,
    exponentials,
    combination_scratch,
    generator_scratch,
    exponential_scratch,
)
    for direction in eachindex(output_cotangents)
        output_cotangent = output_cotangents[direction]

        # V = E*U gives dU = E' * dV.
        mul!(
            generator_scratch[direction],
            exponentials[direction]',
            output_cotangent,
        )
        Gaugefields.add_U!(
            link_cotangents[direction], generator_scratch[direction],
        )

        # exp_ta_pullback! uses its bilinear trace convention, hence U*dV'.
        mul!(
            combination_scratch[direction],
            links[direction],
            output_cotangent',
        )
        LatticeMatrices.exp_ta_pullback!(
            _lcnn_field_data(exponential_scratch[direction]),
            _lcnn_field_data(combination_scratch[direction]),
            _lcnn_field_data(combinations[direction]),
            1,
        )
        Gaugefields.Traceless_antihermitian!(
            generator_scratch[direction], exponential_scratch[direction],
        )

        for channel in eachindex(channels)
            Gaugefields.add_U!(
                channel_cotangents[channel],
                -weights[direction, channel],
                generator_scratch[direction],
            )
            if weight_cotangent !== nothing
                weight_cotangent[direction, channel] -= real(dot(
                    _lcnn_field_data(generator_scratch[direction]),
                    _lcnn_field_data(channels[channel]),
                ))
            end
        end
        Gaugefields.clear_U!(output_cotangent)
    end
    return nothing
end

function _lcnn_zero_parameters!(parameters::AbstractArray)
    fill!(parameters, zero(eltype(parameters)))
    return parameters
end

function _lcnn_zero_parameters!(parameters::NamedTuple)
    foreach(_lcnn_zero_parameters!, values(parameters))
    return parameters
end

function _lcnn_zero_parameters!(parameters::Tuple)
    foreach(_lcnn_zero_parameters!, parameters)
    return parameters
end

@inline function _lcnn_parameter_objective(
    parameters, action, links, workspace,
)
    return action(links, parameters, workspace)
end

"""
    LCNN.parameter_gradient!(gradient, action, U, parameters, workspace,
                             shadow_workspace)

Compute the gradient of a scalar `LCNNAction` with respect to every parameter.
"""
function LCNN.parameter_gradient!(
    gradient,
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters,
    workspace::LCNN.ModelWorkspace{T},
    shadow_workspace::LCNN.ModelWorkspace{T},
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    LCNN.validate_parameters(action, parameters)
    _lcnn_zero_parameters!(gradient)
    Enzyme.remake_zero!(shadow_workspace)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_lcnn_parameter_objective),
        Enzyme.Active,
        Enzyme.Duplicated(parameters, gradient),
        Enzyme.Const(action),
        Enzyme.Const(links),
        Enzyme.Duplicated(workspace, shadow_workspace),
    )
    return gradient
end

function LCNN.parameter_gradient!(
    gradient,
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters,
    workspace::LCNN.ModelWorkspace{T}=LCNN.ModelWorkspace(action, links),
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    return LCNN.parameter_gradient!(
        gradient,
        action,
        links,
        parameters,
        workspace,
        Enzyme.make_zero(workspace),
    )
end

function LCNN.parameter_gradient(
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters;
    workspace::LCNN.ModelWorkspace{T}=LCNN.ModelWorkspace(action, links),
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    gradient = Enzyme.make_zero(parameters)
    return LCNN.parameter_gradient!(
        gradient, action, links, parameters, workspace,
    )
end

"""
    LCNN.link_model_pullback!(dU, dparameters, dUnew, model, U, parameters,
                              workspace[, shadow_workspace])

Apply the VJP of a link-valued `LCNNLinkModel`. `dparameters` may be `nothing`
when only the thin-link cotangent is required.
"""
function LCNN.link_model_pullback!(
    input_cotangent::Vector{T},
    parameter_cotangent,
    output_cotangent::Vector{T},
    model::LCNN.LCNNLinkModel{Dim},
    links::Vector{T},
    parameters,
    workspace::LCNN.LinkModelWorkspace{T},
    shadow_workspace::LCNN.LinkModelWorkspace{T},
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    length(input_cotangent) == Dim || throw(DimensionMismatch(
        "LCNN link pullback expects $Dim input cotangents",
    ))
    length(output_cotangent) == Dim || throw(DimensionMismatch(
        "LCNN link pullback expects $Dim output cotangents",
    ))
    LCNN.validate_parameters(model, parameters)
    LCNN.forward_links!(model, links, parameters, workspace)
    Gaugefields.clear_U!.(input_cotangent)
    parameter_cotangent === nothing ||
        _lcnn_zero_parameters!(parameter_cotangent)
    Enzyme.remake_zero!(shadow_workspace)
    for direction in 1:Dim
        Gaugefields.add_U!(
            shadow_workspace.output[direction],
            output_cotangent[direction],
        )
        Gaugefields.set_wing_U!(shadow_workspace.output[direction])
    end

    feature_channels = last(workspace.feature.feature_buffers)
    feature_cotangents = last(shadow_workspace.feature.feature_buffers)
    lexp_weight_cotangent = parameter_cotangent === nothing ? nothing :
        parameter_cotangent.lexp.weight
    _lcnn_lexp_pullback!(
        input_cotangent,
        feature_cotangents,
        lexp_weight_cotangent,
        shadow_workspace.output,
        links,
        feature_channels,
        parameters.lexp.weight,
        workspace.combinations,
        workspace.exponentials,
        shadow_workspace.combinations,
        shadow_workspace.generators,
        shadow_workspace.exponentials,
    )

    feature_model = model.feature_model
    for layer_index in reverse(eachindex(feature_model.layers))
        layer_parameter_cotangent = parameter_cotangent === nothing ?
            nothing : parameter_cotangent.layers[layer_index]
        _lcnn_lcb_pullback!(
            shadow_workspace.feature.feature_buffers[layer_index],
            input_cotangent,
            layer_parameter_cotangent,
            shadow_workspace.feature.feature_buffers[layer_index + 1],
            feature_model.layers[layer_index],
            links,
            workspace.feature.feature_buffers[layer_index],
            parameters.layers[layer_index],
            workspace.feature.reference,
        )
    end

    plaquette_cotangents = first(shadow_workspace.feature.feature_buffers)
    channel = 1
    for mu in eachindex(links)
        for nu in (mu + 1):length(links)
            Gaugefields.set_wing_U!(plaquette_cotangents[channel])
            _lcnn_plaquette_link_pullback!(
                input_cotangent,
                links,
                plaquette_cotangents[channel],
                mu,
                nu,
                workspace.feature.reference,
            )
            channel += 1
        end
    end
    Gaugefields.clear_U!.(plaquette_cotangents)
    Gaugefields.set_wing_U!.(input_cotangent)
    return parameter_cotangent === nothing ? input_cotangent :
        (; links=input_cotangent, parameters=parameter_cotangent)
end

function LCNN.link_model_pullback!(
    input_cotangent::Vector{T},
    parameter_cotangent,
    output_cotangent::Vector{T},
    model::LCNN.LCNNLinkModel{Dim},
    links::Vector{T},
    parameters,
    workspace::LCNN.LinkModelWorkspace{T},
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    return LCNN.link_model_pullback!(
        input_cotangent,
        parameter_cotangent,
        output_cotangent,
        model,
        links,
        parameters,
        workspace,
        Enzyme.make_zero(workspace),
    )
end

function LCNN.link_model_pullback(
    output_cotangent::Vector{T},
    model::LCNN.LCNNLinkModel{Dim},
    links::Vector{T},
    parameters;
    workspace::LCNN.LinkModelWorkspace{T}=
        LCNN.LinkModelWorkspace(model, links),
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    input_cotangent = [similar(first(links)) for _ in 1:Dim]
    parameter_cotangent = Enzyme.make_zero(parameters)
    return LCNN.link_model_pullback!(
        input_cotangent,
        parameter_cotangent,
        output_cotangent,
        model,
        links,
        parameters,
        workspace,
    )
end

function LCNN.link_model_pullback(
    output_cotangent::Vector{T},
    links::Vector{T},
    cache::LCNN.LCNNLinkSmearingCache,
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    return LCNN.link_model_pullback(
        output_cotangent,
        cache.smearing.model,
        links,
        cache.smearing.parameters;
        workspace=cache.workspace,
    )
end

@inline function _lcnn_link_objective(links, parameters, action, workspace)
    return action(links, parameters, workspace)
end

"""
    LCNN.link_gradient!(dSdu, action, U, parameters, workspace[, dworkspace])

Evaluate the reverse-mode derivative of the scalar `action` with respect to
all complex link-matrix entries.  The flexible Julia model/readout is traced
by Enzyme, while plaquette and one-step transport kernels use the analytic
custom rules in this extension. Longer Wilson lines replay that one-step rule.
`parameters` are treated as constants.
Supplying the zeroable shadow `dworkspace` makes repeated calls reuse all
workspace storage.
"""
function LCNN.link_gradient!(
    gradient::Vector{T},
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters,
    workspace::LCNN.ModelWorkspace{T},
    shadow_workspace::LCNN.ModelWorkspace{T},
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    length(gradient) == Dim || throw(DimensionMismatch(
        "a $Dim-dimensional LCNN link gradient needs $Dim fields, got " *
        "$(length(gradient))",
    ))
    LCNN.validate_parameters(action, parameters)
    Gaugefields.set_wing_U!.(links)
    Gaugefields.clear_U!.(gradient)
    # `ModelWorkspace` has immutable structural fields containing mutable
    # lattice arrays. The shadow was created by `make_zero`, so Enzyme's
    # dedicated reset operation can safely clear those arrays for reuse.
    Enzyme.remake_zero!(shadow_workspace)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_lcnn_link_objective),
        Enzyme.Active,
        Enzyme.Duplicated(links, gradient),
        Enzyme.Const(parameters),
        Enzyme.Const(action),
        Enzyme.Duplicated(workspace, shadow_workspace),
    )
    Gaugefields.set_wing_U!.(gradient)
    return gradient
end

function LCNN.link_gradient!(
    gradient::Vector{T},
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters,
    workspace::LCNN.ModelWorkspace{T},
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    return LCNN.link_gradient!(
        gradient,
        action,
        links,
        parameters,
        workspace,
        Enzyme.make_zero(workspace),
    )
end

function LCNN.link_gradient!(
    gradient::Vector{T},
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters,
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    return LCNN.link_gradient!(
        gradient,
        action,
        links,
        parameters,
        LCNN.ModelWorkspace(action, links),
    )
end

function LCNN.link_gradient(
    action::LCNN.LCNNAction,
    links::Vector{T},
    parameters;
    workspace::LCNN.ModelWorkspace{T}=LCNN.ModelWorkspace(action, links),
) where {NC,Dim,T<:Gaugefields.AbstractGaugefields{NC,Dim}}
    gradient = [similar(first(links)) for _ in 1:Dim]
    return LCNN.link_gradient!(gradient, action, links, parameters, workspace)
end

@inline _enzyme_workspace(x) = x
@static if VERSION >= v"1.12"
    @inline _enzyme_workspace(x::AbstractVector) = Tuple(x)
end

@inline _lm_links(fields) = getproperty.(fields, :U)

function _fold_and_zero_gradient!(field::Gaugefields_4D_MPILattice)
    for dimension in length(field.U.PN):-1:1
        fold_halo_dim_to_core_grad!(field.U, dimension)
    end
    zero_halo_region!(field.U)
    return nothing
end

"""An Enzyme-differentiated scalar potential used by the MD driver."""
struct EnzymeMDAction{F,A}
    potential::F
    arguments::A
    num_temps::Int
end

struct EnzymeMDWorkspace{G,T,DT,P}
    gradient::G
    temps::T
    dtemps::DT
    projection::P
end

function enzyme_md_action(potential, arguments...; num_temps::Integer=0)
    num_temps >= 0 || throw(ArgumentError(
        "num_temps must be nonnegative; got $num_temps",
    ))
    return EnzymeMDAction(potential, arguments, Int(num_temps))
end

function md_action_workspace(
    action::EnzymeMDAction,
    U::Vector{T},
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}
    length(U) == 4 || throw(ArgumentError(
        "Enzyme MD currently supports four-dimensional configurations only",
    ))
    Gaugefields.gauge_halo_width(U) >= 1 || throw(ArgumentError(
        "Enzyme MD requires halo >= 1",
    ))

    gradient = [similar(U[1]) for _ in 1:4]
    if iszero(action.num_temps)
        temps = nothing
        dtemps = nothing
    else
        temps = [similar(U[1]) for _ in 1:action.num_temps]
        dtemps = [similar(U[1]) for _ in 1:action.num_temps]
    end
    return EnzymeMDWorkspace(gradient, temps, dtemps, similar(U[1]))
end

function md_action_workspace(action::EnzymeMDAction, U)
    throw(ArgumentError(
        "Enzyme MD currently supports four-dimensional LatticeMatrices " *
        "configurations only; got $(typeof(U))",
    ))
end

function md_potential(action::EnzymeMDAction, U, workspace::EnzymeMDWorkspace)
    Gaugefields.set_wing_U!.(U)
    links = _lm_links(U)
    value = if workspace.temps === nothing
        action.potential(links..., action.arguments...)
    else
        Gaugefields.clear_U!.(workspace.temps)
        action.potential(
            links...,
            action.arguments...,
            _lm_links(workspace.temps),
        )
    end
    value isa Real || throw(ArgumentError(
        "an Enzyme MD potential must return a real scalar; got $(typeof(value))",
    ))
    return value
end

function md_force!(
    force,
    action::EnzymeMDAction,
    U,
    workspace::EnzymeMDWorkspace,
)
    length(force) == 4 == length(U) || throw(ArgumentError(
        "Enzyme MD requires four gauge links and four force fields",
    ))
    Gaugefields.set_wing_U!.(U)
    Gaugefields.clear_U!.(workspace.gradient)
    constant_arguments = map(nodiff, action.arguments)
    LatticeMatrices.Enzyme_derivative!(
        action.potential,
        _lm_links(U)...,
        _lm_links(workspace.gradient)...,
        constant_arguments...;
        temp=workspace.temps === nothing ? nothing : _lm_links(workspace.temps),
        dtemp=workspace.dtemps === nothing ? nothing : _lm_links(workspace.dtemps),
    )

    for direction in eachindex(U)
        mul!(
            workspace.projection,
            U[direction],
            workspace.gradient[direction]',
        )
        Gaugefields.clear_U!(force[direction])
        Gaugefields.Traceless_antihermitian_add!(
            force[direction],
            0.5,
            workspace.projection,
        )
    end
    return nothing
end

function Wiltinger_U!(U::T) where {NC,T<:Gaugefields_4D_MPILattice{NC}}
    Wiltinger!(U.U)
end

function Wiltinger_derivative!(
    func,
    U::Vector{T},
    dfdU::Vector{T}, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    Enzyme.API.strictAliasing!(false)

    # Primary variable: always differentiated
    annU = enzyme_duplicated(
        _enzyme_workspace(U),
        _enzyme_workspace(dfdU),
    )

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU,
            ann_args...
        )
    else
        Gaugefields.clear_U!.(temp)
        Gaugefields.clear_U!.(dtemp)
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU,
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
            ann_args...,
            enzyme_duplicated(
                _enzyme_workspace(temp),
                _enzyme_workspace(dtemp),
            )
        )
    end

    # Convert real/imaginary gradients to Wirtinger derivatives
    Wiltinger_U!.(dfdU)

    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U1::T,
    U2::T,
    U3::T,
    U4::T,
    dfdU1::T,
    dfdU2::T,
    dfdU3::T,
    dfdU4::T, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    #println("Enzyme_derivative! in Gaugefields.jl")
    Enzyme.API.strictAliasing!(false)

    # Primary variables: always differentiated
    annU1 = enzyme_duplicated(U1, dfdU1)
    annU2 = enzyme_duplicated(U2, dfdU2)
    annU3 = enzyme_duplicated(U3, dfdU3)
    annU4 = enzyme_duplicated(U4, dfdU4)

    # Convert additional arguments
    ann_args = map(toann, args)

    # Call Enzyme
    if temp === nothing
        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),     # function object is always treated as read-only
            Active,          # return value is a real scalar
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...
        )
    else
        Gaugefields.clear_U!.(temp)
        Gaugefields.clear_U!.(dtemp)

        result = Enzyme.autodiff(
            Reverse,
            Enzyme.Const(func),
            Active,
            annU1,
            annU2,
            annU3,
            annU4,
            ann_args...,
            enzyme_duplicated(
                _enzyme_workspace(temp),
                _enzyme_workspace(dtemp),
            )
            #ann_args..., DuplicatedNoNeed(temp, dtemp)
        )
    end


    _fold_and_zero_gradient!(dfdU1)
    _fold_and_zero_gradient!(dfdU2)
    _fold_and_zero_gradient!(dfdU3)
    _fold_and_zero_gradient!(dfdU4)


    # Gradients of Active scalar arguments are returned by Enzyme
    return result
end

function Enzyme_derivative!(
    func,
    U::Vector{T},
    dfdU::Vector{T}, args...;
    temp=nothing,
    dtemp=nothing
) where {NC,T<:Gaugefields_4D_MPILattice{NC}}

    error("Enzyme_derivative! does not support Vector U input. Please define a function that takes U1, U2, U3, U4 as separate arguments and run autodiff on that.")

end
export Enzyme_derivative!

end
