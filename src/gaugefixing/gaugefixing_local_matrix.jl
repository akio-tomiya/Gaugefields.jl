@inline function _set_identity!(matrix, ::Val{NC}) where {NC}
    T = eltype(matrix)
    @inbounds for j in 1:NC
        for i in 1:NC
            matrix[i, j] = ifelse(i == j, one(T), zero(T))
        end
    end
    return nothing
end

@inline function _su2_subgroup_hit!(
    G_tmp,
    w_x,
    M_tmp,
    A_tmp,
    omega,
    coeff2,
    coeff3,
    ::Val{NC},
) where {NC}
    T = eltype(G_tmp)
    RT = typeof(real(zero(T)))
    one_rt = one(RT)
    eps_rt = eps(RT)
    _set_identity!(G_tmp, Val(NC))

    @inbounds for hit_color in 1:(NC * (NC - 1) ÷ 2)
        _set_identity!(A_tmp, Val(NC))
        i1, i2 = get_SU2_index(NC, hit_color)

        norm2 = abs2(conj(w_x[i1, i1]) + w_x[i2, i2]) +
                abs2(conj(w_x[i2, i1]) - w_x[i1, i2])
        if norm2 > eps_rt
            invnorm = inv(sqrt(norm2))
            su2_11 = invnorm * (conj(w_x[i1, i1]) + w_x[i2, i2])
            su2_12 = invnorm * (-w_x[i1, i2] + conj(w_x[i2, i1]))
            su2_21 = invnorm * (conj(w_x[i1, i2]) - w_x[i2, i1])
            su2_22 = invnorm * (w_x[i1, i1] + conj(w_x[i2, i2]))

            if omega > one_rt
                m11 = su2_11 - one(T)
                m12 = su2_12
                m21 = su2_21
                m22 = su2_22 - one(T)

                m2_11 = m11 * m11 + m12 * m21
                m2_12 = m11 * m12 + m12 * m22
                m2_21 = m21 * m11 + m22 * m21
                m2_22 = m21 * m12 + m22 * m22

                m3_11 = m2_11 * m11 + m2_12 * m21
                m3_12 = m2_11 * m12 + m2_12 * m22
                m3_21 = m2_21 * m11 + m2_22 * m21
                m3_22 = m2_21 * m12 + m2_22 * m22

                su2_11 *= omega
                su2_12 *= omega
                su2_21 *= omega
                su2_22 *= omega
                su2_11 += coeff2 * m2_11
                su2_12 += coeff2 * m2_12
                su2_21 += coeff2 * m2_21
                su2_22 += coeff2 * m2_22
                su2_11 += coeff3 * m3_11
                su2_12 += coeff3 * m3_12
                su2_21 += coeff3 * m3_21
                su2_22 += coeff3 * m3_22

                column_norm1 = sqrt(abs2(su2_11) + abs2(su2_21))
                if column_norm1 > eps_rt
                    su2_11 /= column_norm1
                    su2_21 /= column_norm1
                    projection =
                        conj(su2_11) * su2_12 + conj(su2_21) * su2_22
                    su2_12 -= projection * su2_11
                    su2_22 -= projection * su2_21
                    column_norm2 = sqrt(abs2(su2_12) + abs2(su2_22))
                    if column_norm2 > eps_rt
                        su2_12 /= column_norm2
                        su2_22 /= column_norm2
                    else
                        su2_12 = -conj(su2_21)
                        su2_22 = conj(su2_11)
                    end
                else
                    su2_11 = one(T)
                    su2_12 = zero(T)
                    su2_21 = zero(T)
                    su2_22 = one(T)
                end
            end

            A_tmp[i1, i1] = su2_11
            A_tmp[i1, i2] = su2_12
            A_tmp[i2, i1] = su2_21
            A_tmp[i2, i2] = su2_22
        end

        mul!(M_tmp, A_tmp, G_tmp)
        for jc in 1:NC
            for ic in 1:NC
                G_tmp[ic, jc] = M_tmp[ic, jc]
            end
        end
    end
    return nothing
end

@inline function _steepest_descent_transform!(
    G_tmp,
    Um_tmp,
    Up_tmp,
    Δ_tmp,
    Δ2,
    overrelax,
    ::Val{NC},
) where {NC}
    T = eltype(G_tmp)
    RT = typeof(real(zero(T)))
    _set_identity!(G_tmp, Val(NC))

    numerator = zero(RT)
    @inbounds for row in 1:NC
        for k in 1:NC
            numerator += real(Δ_tmp[row, k] * Um_tmp[k, row])
        end
    end

    @inbounds for col in 1:NC
        for row in 1:NC
            value = zero(T)
            for k in 1:NC
                value += Δ_tmp[row, k] * Δ_tmp[k, col]
            end
            Δ2[row, col] = value
        end
    end

    denominator = zero(RT)
    @inbounds for row in 1:NC
        for k in 1:NC
            denominator += real(Δ2[row, k] * Up_tmp[k, row])
        end
    end

    if abs(denominator) > eps(RT)
        alpha = -numerator / denominator
        scale = convert(RT, overrelax) * alpha
        @inbounds for col in 1:NC
            for row in 1:NC
                G_tmp[row, col] += scale * Δ_tmp[row, col]
            end
        end
    end
    return nothing
end
