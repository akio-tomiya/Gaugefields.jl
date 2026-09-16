using JACC
JACC.@init_backend

using Enzyme
using Gaugefields
using HDF5
using Optimisers
using Random

const LCNN = Gaugefields.LCNN

function selected_samples()
    limit = parse(Int, get(ENV, "LCNN_MAX_SAMPLES", "0"))
    return iszero(limit) ? :all : (1:limit)
end

function main(train_path, validation_path, test_path)
    samples = selected_samples()
    train = LCNN.read_favoni2022_dataset(
        train_path; target="trW_1x2", samples,
    )
    validation = LCNN.read_favoni2022_dataset(
        validation_path; target="trW_1x2", samples,
    )
    test = LCNN.read_favoni2022_dataset(
        test_path; target="trW_1x2", samples,
    )

    train.lattice == (8, 8) || @warn(
        "the published D2 W1x2 run used an 8 by 8 lattice",
        lattice=train.lattice,
    )
    action = LCNN.favoni2022_wilson_1x2_small(
        ; convention=:favoni_prl,
    )
    parameters = LCNN.initial_parameters(
        MersenneTwister(1234), action, Float32,
    )
    config = LCNN.TrainingConfig(
        ; max_epochs=20,
        batch_size=50,
        learning_rate=3.0f-3,
        weight_decay=0.0f0,
        amsgrad=true,
        patience=5,
        min_delta=0.0f0,
        seed=1234,
    )

    result = LCNN.fit!(
        action,
        parameters,
        train,
        validation;
        config,
        callback=record -> println(
            "epoch ", record.epoch,
            ": train=", record.training_loss,
            ", validation=", record.validation_loss,
        ),
    )

    println("best epoch = ", result.best_epoch)
    println("best validation site MSE = ", result.best_validation_loss)
    println("test site MSE = ", LCNN.evaluate_dataset(
        action, result.parameters, test,
    ))
    println("test global-average MSE = ", LCNN.evaluate_dataset(
        action, result.parameters, test; global_average=true,
    ))
end

length(ARGS) == 3 || error(
    "usage: julia --project favoni2022_hdf5_training.jl " *
    "TRAIN.hdf5 VAL.hdf5 TEST.hdf5",
)
main(ARGS...)
