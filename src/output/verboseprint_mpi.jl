module Verboseprint_mpi


struct Verbose_print
    level::Int8
    fp::Union{Nothing,IOStream}
    myid::Int64

    function Verbose_print(level; myid = 0, filename = nothing)
        if filename == nothing
            fp = nothing
        else
            if myid == 0
                fp = open(filename, "w")
            else
                fp = nothing
            end
        end
        return new(level, fp, myid)
    end
end

function println_verbose_level1(v::Verbose_print, val...)
    if v.myid == 0 && v.level >= 1
        println(val...)
        if v.fp != nothing
            println(v.fp, val...)
        end
    end
end

function println_verbose_level2(v::Verbose_print, val...)
    if v.myid == 0 && v.level >= 2
        println(val...)
        if v.fp != nothing
            println(v.fp, val...)
        end
    end
end

function println_verbose_level3(v::Verbose_print, val...)
    if v.myid == 0 && v.level >= 3
        println(val...)
        if v.fp != nothing
            println(v.fp, val...)
        end
    end
end

end
