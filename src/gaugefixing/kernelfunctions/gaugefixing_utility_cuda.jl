import StaticArrays: @MMatrix



function cudakernel_SU2_subgroup_hit(u1, u2, u3, u4, g, parity::Int, overrelax::Float64, coeff2::Float64, coeff3::Float64, NC, blockinfo)

    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)

    #share_w_x = CuDynamicSharedArray(ComplexF64, (3, 3, nthreads))
    #w_x = view(share_w_x, :,:,b)

    w_x = @MMatrix zeros(ComplexF64, 3, 3)
    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    M_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    A_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    

    #G_tmp .= ComplexF64(0)
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    
    # Do only even or odd parity site each time
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)    
    
    parity_check =  (ix+iy+iz+it) % 2 # To be checked
    
    if parity_check == parity

        for ic in 1:3
            for jc in 1:3
                w_x[ic, jc] = 0
            end
        end

        bshifted_1, rshifted_1 = shiftedindex(b, r, (-1,0,0,0), blockinfo)
        bshifted_2, rshifted_2 = shiftedindex(b, r, (0,-1,0,0), blockinfo)
        bshifted_3, rshifted_3 = shiftedindex(b, r, (0,0,-1,0), blockinfo)
        bshifted_4, rshifted_4 = shiftedindex(b, r, (0,0,0,-1), blockinfo)

        # Compute w = U_μ(x) + U_μ^†( x - μ)
        for ic in 1:3
            for jc in 1:3
                w_x[ic, jc]  = (  u1[ic, jc, b, r] + conj(u1[jc, ic, bshifted_1, rshifted_1])
                                + u2[ic, jc, b, r] + conj(u2[jc, ic, bshifted_2, rshifted_2])
                                + u3[ic, jc, b, r] + conj(u3[jc, ic, bshifted_3, rshifted_3])
                                + u4[ic, jc, b, r] + conj(u4[jc, ic, bshifted_4, rshifted_4]))

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
                #coeff2 = overrelax * (overrelax - 1) / 2
                #coeff3 = overrelax * (overrelax - 1) * (overrelax - 2) / 6

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
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end

    end

    return
end


function cudakernel_mino_method(u1, u2, u3, u4, Δ, g, parity::Int, overrelax::Float64, NC, blockinfo)

    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)

    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Um_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Up_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Δ_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Δ2 = @MMatrix zeros(ComplexF64, 3, 3)
    
    
    
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    # Do only even or odd parity site each time
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
    parity_check =  (ix+iy+iz+it) % 2 # To be checked
    if parity_check == parity

        bshifted_1, rshifted_1 = shiftedindex(b, r, (-1,0,0,0), blockinfo)
        bshifted_2, rshifted_2 = shiftedindex(b, r, (0,-1,0,0), blockinfo)
        bshifted_3, rshifted_3 = shiftedindex(b, r, (0,0,-1,0), blockinfo)
        bshifted_4, rshifted_4 = shiftedindex(b, r, (0,0,0,-1), blockinfo)

        for ic in 1:3
            for jc in 1:3
                Um_tmp[ic, jc]  = u1[ic, jc, b, r] - u1[ic, jc, bshifted_1, rshifted_1]
                Um_tmp[ic, jc] += u2[ic, jc, b, r] - u2[ic, jc, bshifted_2, rshifted_2]
                Um_tmp[ic, jc] += u3[ic, jc, b, r] - u3[ic, jc, bshifted_3, rshifted_3]
                Um_tmp[ic, jc] += u4[ic, jc, b, r] - u4[ic, jc, bshifted_4, rshifted_4]

                Up_tmp[ic, jc]  = u1[ic, jc, b, r] + u1[ic, jc, bshifted_1, rshifted_1]
                Up_tmp[ic, jc] += u2[ic, jc, b, r] + u2[ic, jc, bshifted_2, rshifted_2]
                Up_tmp[ic, jc] += u3[ic, jc, b, r] + u3[ic, jc, bshifted_3, rshifted_3]
                Up_tmp[ic, jc] += u4[ic, jc, b, r] + u4[ic, jc, bshifted_4, rshifted_4]

                
                Δ_tmp[ic, jc] = Δ[ic, jc, b, r]
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
        



        # Step 6: Orthonormalize G_tmp (e.g., with your 3x3 Gram-Schmidt)
        #gramschmidt!(G_tmp)

        
        for i in 1:3
            # Step 1: Subtract projections of previous columns
            for j in 1:i-1
                # Compute the inner product between column i and column j (Hermitian)
                dot_prod = 0.0 + 0.0im  # initialize complex dot product
                for k in 1:3
                    dot_prod += conj(G_tmp[k,j]) * G_tmp[k,i]  # Hermitian inner product
                end
                
                # Subtract projection from column i
                for k in 1:3
                    G_tmp[k,i] -= dot_prod * G_tmp[k,j]
                end
            end
            
            # Step 2: Normalize column i
            norm_val = 0.0 + 0.0im  # initialize complex norm value
            for k in 1:3
                norm_val += conj(G_tmp[k,i]) * G_tmp[k,i]  # norm is the inner product of column with itself
            end
            norm_val = sqrt(real(norm_val))  # We use the real part of the norm

            # Step 3: Normalize the column
            for k in 1:3
                G_tmp[k,i] /= norm_val
            end
        end
    end
    
    for ic in 1:3
        for jc in 1:3
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end
    end

    
    return
end


function cudakernel_SU2_subgroup_hit_D3(u1, u2, u3, u4, g, parity::Int, overrelax::Float64, coeff2::Float64, coeff3::Float64, NC, blockinfo)

    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)

    w_x = @MMatrix zeros(ComplexF64, 3, 3)
    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    M_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    A_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    

    #G_tmp .= ComplexF64(0)
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    
    # Do only even or odd parity site each time
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)    
    
    parity_check =  (ix+iy+iz+it) % 2 # To be checked
    
    if parity_check == parity

        for ic in 1:3
            for jc in 1:3
                w_x[ic, jc] = 0
            end
        end

        bshifted_1, rshifted_1 = shiftedindex(b, r, (-1,0,0,0), blockinfo)
        bshifted_2, rshifted_2 = shiftedindex(b, r, (0,-1,0,0), blockinfo)
        bshifted_3, rshifted_3 = shiftedindex(b, r, (0,0,-1,0), blockinfo)
        

        # Compute w = U_μ(x) + U_μ^†( x - μ)
        for ic in 1:3
            for jc in 1:3
                w_x[ic, jc]  = (  u1[ic, jc, b, r] + conj(u1[jc, ic, bshifted_1, rshifted_1])
                                + u2[ic, jc, b, r] + conj(u2[jc, ic, bshifted_2, rshifted_2])
                                + u3[ic, jc, b, r] + conj(u3[jc, ic, bshifted_3, rshifted_3]))

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
            
            
            if overrelax > 1.0
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
                #coeff2 = overrelax * (overrelax - 1) / 2
                #coeff3 = overrelax * (overrelax - 1) * (overrelax - 2) / 6

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
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end

    end

    return
end


function cudakernel_mino_method_D3(u1, u2, u3, u4, Δ, g, parity::Int, overrelax::Float64, NC, blockinfo)

    b = Int(CUDA.threadIdx().x)
    r = Int(CUDA.blockIdx().x)

    G_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Um_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Up_tmp = @MMatrix zeros(ComplexF64, 3, 3)

    Δ_tmp = @MMatrix zeros(ComplexF64, 3, 3)
    Δ2 = @MMatrix zeros(ComplexF64, 3, 3)
    
    
    
    @inbounds for ic in 1:3
        G_tmp[ic,ic] = 1f0 + 0im
    end
    # Do only even or odd parity site each time
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
    parity_check =  (ix+iy+iz+it) % 2 # To be checked
    if parity_check == parity

        bshifted_1, rshifted_1 = shiftedindex(b, r, (-1,0,0,0), blockinfo)
        bshifted_2, rshifted_2 = shiftedindex(b, r, (0,-1,0,0), blockinfo)
        bshifted_3, rshifted_3 = shiftedindex(b, r, (0,0,-1,0), blockinfo)

        for ic in 1:3
            for jc in 1:3
                Um_tmp[ic, jc]  = u1[ic, jc, b, r] - u1[ic, jc, bshifted_1, rshifted_1]
                Um_tmp[ic, jc] += u2[ic, jc, b, r] - u2[ic, jc, bshifted_2, rshifted_2]
                Um_tmp[ic, jc] += u3[ic, jc, b, r] - u3[ic, jc, bshifted_3, rshifted_3]

                Up_tmp[ic, jc]  = u1[ic, jc, b, r] + u1[ic, jc, bshifted_1, rshifted_1]
                Up_tmp[ic, jc] += u2[ic, jc, b, r] + u2[ic, jc, bshifted_2, rshifted_2]
                Up_tmp[ic, jc] += u3[ic, jc, b, r] + u3[ic, jc, bshifted_3, rshifted_3]

                
                Δ_tmp[ic, jc] = Δ[ic, jc, b, r]
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
        



        # Step 6: Orthonormalize G_tmp (e.g., with your 3x3 Gram-Schmidt)
        #gramschmidt!(G_tmp)

        
        for i in 1:3
            # Step 1: Subtract projections of previous columns
            for j in 1:i-1
                # Compute the inner product between column i and column j (Hermitian)
                dot_prod = 0.0 + 0.0im  # initialize complex dot product
                for k in 1:3
                    dot_prod += conj(G_tmp[k,j]) * G_tmp[k,i]  # Hermitian inner product
                end
                
                # Subtract projection from column i
                for k in 1:3
                    G_tmp[k,i] -= dot_prod * G_tmp[k,j]
                end
            end
            
            # Step 2: Normalize column i
            norm_val = 0.0 + 0.0im  # initialize complex norm value
            for k in 1:3
                norm_val += conj(G_tmp[k,i]) * G_tmp[k,i]  # norm is the inner product of column with itself
            end
            norm_val = sqrt(real(norm_val))  # We use the real part of the norm

            # Step 3: Normalize the column
            for k in 1:3
                G_tmp[k,i] /= norm_val
            end
        end
    end
    
    for ic in 1:3
        for jc in 1:3
            g[ic, jc, b, r] = G_tmp[ic, jc]
        end
    end

    
    return
end