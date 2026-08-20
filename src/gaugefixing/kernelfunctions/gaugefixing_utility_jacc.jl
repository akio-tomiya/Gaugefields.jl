import StaticArrays: @MMatrix


function jacckernel_SU2_subgroup_hit!(i, g, W, dindexer, parity::Int, overrelax::Float64, coeff2::Float64, coeff3::Float64, ::Val{NC}, ::Val{nw}) where {NC,nw}

    indices = delinearize(dindexer, i, nw)

    w_x = @MMatrix zeros(ComplexF64, 3, 3)
    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    M_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    A_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    

    #G_tmp .= ComplexF64(0)
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    
    # Do only even or odd parity site each time 
    parity_check =  sum(indices) % 2 # nw is implicitly considered here
    
    if parity_check == parity

        #w_x = W[1,1, indices...]
        for ic in 1:3
            for jc in 1:3
                w_x[ic, jc] = W[ic, jc, indices...]
            end
        end    
        
        # Reunitarisation process
        for hit_color::Int in 1:Int(NC*(NC-1)/2) 

            A_tmp .= ComplexF64(0)
            @inbounds for ic in 1:3
                A_tmp[ic,ic] = 1f0 + 0im
            end

                                
            i1, i2 = get_SU2_index(NC, hit_color)
            nor_factor = 1/sqrt(abs( conj(w_x[i1,i1]) + w_x[i2,i2] )^2 + abs( conj(w_x[i2,i1]) - w_x[i1,i2] )^2 )
            
            
            if overrelax > 1.0  # TO DO: should be checked with nowing_mpi 
                #overrelax = BigFloat(overrelax)

                # assigning SU(2) ovr matrix
                su2_11 =  nor_factor * ( conj(w_x[i1, i1])    + w_x[i2,i2]) 
                su2_12 =  nor_factor * (-w_x[i1, i2]          + conj(w_x[i2,i1]))
                su2_21 =  nor_factor * ( conj(w_x[i1,i2])     - w_x[i2, i1])
                su2_22 =  nor_factor * ( w_x[i1, i1]          + conj(w_x[i2,i2])) 
                
                # over
                # Step 1: M = su2 - I
                m11 = su2_11 - 1
                m12 = su2_12
                m21 = su2_21
                m22 = su2_22 - 1

                # Step 2: M^2 = M * M
                m2_11 = m11*m11 + m12*m21
                m2_12 = m11*m12 + m12*m22
                m2_21 = m21*m11 + m22*m21
                m2_22 = m21*m12 + m22*m22

                # Step 3: M^3 = M^2 * M
                m3_11 = m2_11*m11 + m2_12*m21
                m3_12 = m2_11*m12 + m2_12*m22
                m3_21 = m2_21*m11 + m2_22*m21
                m3_22 = m2_21*m12 + m2_22*m22

                # Step 4: scalar coefficients (no gamma!)
                # coeff2 = overrelax * (overrelax - 1) / 2
                # coeff3 = overrelax * (overrelax - 1) * (overrelax - 2) / 6

                # Step 5: scale original su2
                su2_11 *= overrelax
                su2_12 *= overrelax
                su2_21 *= overrelax
                su2_22 *= overrelax

                # Step 6: add coeff2 * M^2
                su2_11 += coeff2 * m2_11
                su2_12 += coeff2 * m2_12
                su2_21 += coeff2 * m2_21
                su2_22 += coeff2 * m2_22

                # Step 7: add coeff3 * M^3
                su2_11 += coeff3 * m3_11
                su2_12 += coeff3 * m3_12
                su2_21 += coeff3 * m3_21
                su2_22 += coeff3 * m3_22

                #########
                
                # gramschmidt
                # First column normalization
                norm1 = sqrt(abs2(su2_11) + abs2(su2_21))
                su2_11 /= norm1
                su2_21 /= norm1

                # Dot product with Hermitian (conj) for projection
                dot = conj(su2_11)*su2_12 + conj(su2_21)*su2_22

                # Subtract projection
                su2_12 -= dot * su2_11
                su2_22 -= dot * su2_21

                # Normalize second column
                norm2 = sqrt(abs2(su2_12) + abs2(su2_22))
                su2_12 /= norm2
                su2_22 /= norm2

                A_tmp[i1, i1] =  su2_11 
                A_tmp[i1, i2] =  su2_12 
                A_tmp[i2, i1] =  su2_21 
                A_tmp[i2, i2] =  su2_22 
                
            else
                # no overrelax, ovr = 1.0
                A_tmp[i1, i1] =  nor_factor * ( conj(w_x[i1, i1])    + w_x[i2,i2])
                A_tmp[i1, i2] =  nor_factor * (-w_x[i1, i2]          + conj(w_x[i2,i1]))
                A_tmp[i2, i1] =  nor_factor * ( conj(w_x[i1,i2])     - w_x[i2, i1])
                A_tmp[i2, i2] =  nor_factor * ( w_x[i1, i1]          + conj(w_x[i2,i2]))
            end
            

            mul!(M_tmp, A_tmp, G_tmp)
            for ic in 1:3
                for jc in 1:3
                    G_tmp[ic, jc] = M_tmp[ic, jc]
                end
            end

        end
    end
    
    for ic in 1:3
        for jc in 1:3
            g[ic, jc, indices...] = G_tmp[ic, jc]
        end

    end

    return
end


function jacckernel_mino_method!(i, dindexer, g, Um, Up, Δ, parity::Int, overrelax::Float64, ::Val{nw}) where {nw}

    indices = delinearize(dindexer, i, nw)

    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Um_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Up_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Δ_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Δ2 = @MMatrix zeros(ComplexF64, 3, 3)
    
    
    
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    # Do only even or odd parity site each time
    parity_check =  parity_check =  sum(indices) % 2 # nw is implicitly considered here
    @inbounds begin
        if parity_check == parity

            for ic in 1:3
                for jc in 1:3
                    Um_tmp[ic, jc]  = Um[ic, jc, indices...]
                    Up_tmp[ic, jc]  = Up[ic, jc, indices...]
                    Δ_tmp[ic, jc] = Δ[ic, jc, indices...]
                end
            end
            
            # Step 1: Compute tr(Δ * Um)
            num = 0.0
            for i in 1:3
                for k in 1:3
                    num += real(Δ_tmp[i,k] * Um_tmp[k,i])
                end
            end

            # Step 2: Compute Δ² = Δ * Δ
            
            for i in 1:3
                for j in 1:3
                    Δ2[i,j] = 0.0 + 0.0im
                    for k in 1:3
                        Δ2[i,j] += Δ_tmp[i,k] * Δ_tmp[k,j]
                    end
                end
            end

            # Step 3: Compute tr(Δ² * Up)
            denom = 0.0
            for i in 1:3
                for k in 1:3
                    denom += real(Δ2[i,k] * Up_tmp[k,i])
                end
            end

            # Step 4: Compute alpha
            α = -num / denom
            
            
            # Step 5: Update G_tmp
            for i in 1:3
                for j in 1:3
                    G_tmp[i,j] += overrelax * α * Δ_tmp[i,j]
                end
            end
            
        end
    
        for ic in 1:3
            for jc in 1:3
                g[ic, jc, indices...] = G_tmp[ic, jc]
            end
        end

    end

    
    return
end
