using Gaugefields
using Test

function fill_ildg_test_field!(U)
    NX, NY, NZ, NT = U[1].NX, U[1].NY, U[1].NZ, U[1].NT
    NC = U[1].NC
    count = 0
    for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX
        for μ = 1:4, ic2 = 1:NC, ic1 = 1:NC
            count += 1
            U[μ][ic2, ic1, ix, iy, iz, it] =
                count / 997 - im * count / 991
        end
    end
    return U
end

function physical_ildg_values(U)
    NX, NY, NZ, NT = U[1].NX, U[1].NY, U[1].NZ, U[1].NT
    NC = U[1].NC
    return [
        U[μ][ic2, ic1, ix, iy, iz, it]
        for it = 1:NT, iz = 1:NZ, iy = 1:NY, ix = 1:NX,
            μ = 1:4, ic2 = 1:NC, ic1 = 1:NC
    ]
end

@testset "ILDG contiguous local-volume reader" begin
    global_size = (4, 4, 4, 4)
    fields_per_site = 3
    decompositions = (
        ((4, 4, 4, 4), (0, 0, 0, 0)),
        ((4, 4, 4, 2), (0, 0, 0, 2)),
        ((4, 4, 2, 2), (0, 0, 2, 1)),
        ((4, 2, 2, 2), (0, 1, 1, 1)),
        ((2, 2, 2, 2), (1, 1, 1, 1)),
    )

    mktempdir() do dir
        for precision in (64, 32)
            F = precision == 64 ? Float64 : Float32
            number_of_sites = prod(global_size)
            values = Complex{F}[
                Complex{F}(site + field / 10, -site - field / 20)
                for site = 0:(number_of_sites - 1) for field = 1:fields_per_site
            ]
            payload = joinpath(dir, "local-volume-$precision.dat")
            open(payload, "w") do io
                for value in values
                    write(io, hton(real(value)))
                    write(io, hton(imag(value)))
                end
            end

            for (local_size, offset) in decompositions
                expected = Complex{F}[]
                NX, NY, NZ, _ = global_size
                px, py, pz, pt = offset
                for it = 0:(local_size[4] - 1), iz = 0:(local_size[3] - 1),
                    iy = 0:(local_size[2] - 1), ix = 0:(local_size[1] - 1)
                    global_site =
                        (((pt + it) * NZ + (pz + iz)) * NY + (py + iy)) * NX +
                        (px + ix)
                    first_value = global_site * fields_per_site + 1
                    append!(
                        expected,
                        @view(values[first_value:(first_value + fields_per_site - 1)]),
                    )
                end

                result = Vector{Complex{F}}(undef, length(expected))
                bi = Gaugefields.ILDG_format.Binarydata_ILDG(payload, precision)
                try
                    Gaugefields.ILDG_format.read_ildg_local_volume!(
                        result,
                        bi,
                        global_size,
                        local_size,
                        offset,
                        fields_per_site;
                        chunk_bytes=3 * 2 * sizeof(F),
                    )
                    @test bi.count == length(result)
                finally
                    close(bi)
                end
                @test result == expected
            end
        end
    end
end

@testset "ILDG precision and metadata" begin
    L = (2, 2, 2, 2)
    NC = 3
    U = Initialize_Gaugefields(NC, 0, L..., condition="cold")
    fill_ildg_test_field!(U)
    original = physical_ildg_values(U)

    mktempdir() do dir
        for precision in (64, 32)
            filename = joinpath(dir, "roundtrip-$precision.ildg")
            payload = joinpath(dir, "payload-$precision.dat")
            filelist = joinpath(dir, "filelist-$precision.dat")

            save_binarydata(
                U, filename;
                precision,
                tempfile1=payload,
                tempfile2=filelist,
            )

            ildg = ILDG(filename)
            @test length(ildg) == 1
            @test ildg[1]["L"] == L
            @test ildg[1]["NC"] == NC
            @test ildg[1]["precision"] == precision

            restored = Initialize_Gaugefields(NC, 0, L..., condition="cold")
            load_gaugefield!(restored, 1, ildg, L, NC)
            result = physical_ildg_values(restored)
            expected = precision == 64 ? original : ComplexF64.(ComplexF32.(original))
            @test result == expected

            restored_wing = [
                Gaugefields.AbstractGaugefields_module.identityGaugefields_4D_wing(
                    NC,
                    L...,
                    1;
                    verbose_level=0,
                ) for _ = 1:4
            ]
            load_gaugefield!(restored_wing, 1, ildg, L, NC; NDW=1)
            @test physical_ildg_values(restored_wing) == expected
        end

        @test_throws ArgumentError save_binarydata(
            U,
            joinpath(dir, "invalid.ildg");
            precision=16,
            tempfile1=joinpath(dir, "invalid.dat"),
            tempfile2=joinpath(dir, "invalid.list"),
        )
        @test_throws ArgumentError save_binarydata(
            U,
            joinpath(dir, "one-temporary.ildg");
            tempfile1=joinpath(dir, "one-temporary.dat"),
        )

        automatic_dir = joinpath(dir, "automatic")
        mkdir(automatic_dir)
        automatic_file = joinpath(automatic_dir, "configuration.ildg")
        save_configuration(
            automatic_file,
            U;
            format=:ildg,
            precision=32,
        )
        @test readdir(automatic_dir) == ["configuration.ildg"]
        automatic_restored = Initialize_Gaugefields(
            NC,
            0,
            L...;
            condition="cold",
        )
        load_configuration!(automatic_restored, automatic_file; format=:ildg)
        @test physical_ildg_values(automatic_restored) ==
              ComplexF64.(ComplexF32.(original))
    end

    xml = Gaugefields.ILDG_format.ildg_format_xml(L, NC, 32)
    @test occursin("xmlns=\"http://www.lqcd.org/ildg\"", xml)
    @test occursin("<version>1.2</version>", xml)
    @test occursin("<precision>32</precision>", xml)
end
