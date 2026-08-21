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
