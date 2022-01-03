module heatbath_module
using LinearAlgebra

import ..AbstractGaugefields_module:normalize3!,normalizeN!

function SU2update_KP!(Unew,V,beta,NC,temps,ITERATION_MAX = 10^5)
    V0 = temps[1]
    temp = temps[2]

    eps = 0.000000000001

    ρ0 = real(V[1,1]+V[2,2])/2
    ρ1 = -imag(V[1,2]+V[2,1])/2
    #ρ1 = imag(V[1,2]+V[2,1])/2
    ρ2 = real(V[2,1]-V[1,2])/2
    ρ3 = imag(V[2,2]-V[1,1])/2
    ρ = sqrt(ρ0^2+ρ1^2+ρ2^2+ρ3^2)
    #println("R = ",R," ρ ",ρ)
    #println("detV = , ", det(V)," ",ρ0^2+ρ1^2+ρ2^2+ρ3^2)
    detV = V[1,1]*V[2,2] - V[1,2]*V[2,1]
    V0[1,1] = ρ*V[2,2]/detV
    V0[1,2] = -ρ*V[1,2]/detV
    V0[2,1] = -ρ*V[2,1]/detV
    V0[2,2] = ρ*V[1,1]/detV
    #println(V0)
    #V0 = inv(V/ρ)

    #
    #Nc = 2 # Since Ishikawa's book uses 1/g^2 notation.
    #k = (beta/NC)*ρ
    k = 2*(beta/NC)*ρ
    #println("k $k, $ρ")

    
    #k = (beta/2)*ρ

    R = rand() + eps
    Rp = rand() + eps
    X = -log(R)/k
    Xp = -log(Rp)/k
    Rpp = rand()
    C = cos(2pi*Rpp)^2
    A = X*C
    delta = Xp + A
    Rppp = rand()

    a = zeros(Float64,4)
    while(Rppp^2 > 1-0.5*delta)
        R = rand()
        Rp = rand()
        X = -log(R)/k
        Xp = -log(Rp)/k
        Rpp = rand()
        C = cos(2pi*Rpp)^2
        A = X*C
        delta = Xp + A
        Rppp = rand()
        #println(Rppp^2,"\t",1-0.5*delta)
    end
    a[1] = 1-delta


    rr = sqrt(1.0-a[1]^2)
    ϕ = rand()*pi*2.0 # ϕ = [0,2pi]
    cosθ = (rand()-0.5)*2.0 # -1<cosθ<1
    sinθ = sqrt(1-cosθ^2)

    a[2]=rr*cos(ϕ)*sinθ
    a[3]=rr*sin(ϕ)*sinθ
    a[4]=rr*cosθ
    temp[1,1] = a[1]+im*a[4]
    temp[1,2] = a[3]+im*a[2]
    temp[2,1] = -a[3]+im*a[2] 
    temp[2,2] = a[1]-im*a[4]
    mul!(Unew,temp,V0)
    #Unew[:,:] = Unew*V0

    

    α = Unew[1,1]*0.5 + conj(Unew[2,2])*0.5
    β = Unew[2,1]*0.5 - conj(Unew[1,2])*0.5

    detU = abs(α)^2 + abs(β)^2
    Unew[1,1] = α/detU
    Unew[2,1]  = β/detU
    Unew[1,2] = -conj(β)/detU
    Unew[2,2] = conj(α)/detU     
    
end

function SU2update_KP(V,beta,NC,temps,ITERATION_MAX = 10^5)
    #println("V = ",V)
    Unew = zero(V)
    SU2update_KP!(Unew,V,beta,NC,temps,ITERATION_MAX)
    return Unew
end

function SUNupdate_matrix!(u,V,beta,NC,temps2,temps3,ITERATION_MAX)
    UV = temps3[1]
    A = temps3[2]
    AU = temps3[3]
    temp1 = temps2[1]
    temp2 = temps2[2]
    S = temps2[3]
    K = temps2[4]

    for l=1:NC
        #for l=1:2NC
        
        mul!(UV,u,V)
        #UV = u[:,:]*V

        n = rand(1:NC-1)#l
        m = rand(n:NC)
        while(n==m)
            m = rand(n:NC)
        end
        
        #=
        if l < NC
            n = l
            m = l+1
        else
            n = rand(1:NC)#l
            m = rand(1:NC)
            while(n==m)
                m = rand(1:NC)
            end
        end
        =#


        make_submatrix!(S,UV,n,m)
        #S = make_submatrix(UV,n,m)

        #gramschmidt_special!(S)
        project_onto_SU2!(S)

        SU2update_KP!(K,S,beta,NC,[temp1,temp2],ITERATION_MAX)


        make_largematrix!(A,K,n,m,NC)

        mul!(AU,A,u)

        #AU = A*u[:,:]

        u[:,:] .= AU
        #println("det U ",det(AU))

    end

    AU[:,:] .= u #u[:,:]
    normalizeN!(AU)
    u[:,:] .= AU
end

function SU3update_matrix!(u,V,beta,NC,temps2,temps3,ITERATION_MAX)
    UV = temps3[1]
    A = temps3[2]
    AU = temps3[3]
    temp1 = temps2[1]
    temp2 = temps2[2]
    S = temps2[3]
    K = temps2[4]

    #println("#Heatbath for one SU(3) link started")
    for l=1:3
        mul!(UV,u,V)
        #UV = u*V
        #println("UV $UV $V $u")

        if l==1
            n,m = 1,2
        elseif l==2
            n,m = 2,3
        else
            n,m = 1,3

        end

        #S = make_submatrix(UV,n,m)
        make_submatrix!(S,UV,n,m)
        #gramschmidt_special!(S)
        project_onto_SU2!(S)

        #K = SU2update_KP(S,beta,NC,ITERATION_MAX)
        SU2update_KP!(K,S,beta,NC,[temp1,temp2],ITERATION_MAX)


        #A = make_largematrix(K,n,m,NC)
        make_largematrix!(A,K,n,m,NC)

        mul!(AU,A,u)
        #AU = A*u

        u[:,:] .= AU
    end

    AU[:,:] .= u #u[mu][:,:,ix,iy,iz,it]
    normalize3!(AU)
    u[:,:] .= AU
    #u[mu][:,:,ix,iy,iz,it] = AU
end


function project_onto_SU2!(S) # This project onto SU(2) up to normalization.
    #S2 = zeros(ComplexF64,2,2)
    α = S[1,1]*0.5 + conj(S[2,2])*0.5
    β = S[2,1]*0.5 - conj(S[1,2])*0.5
    S[1,1] = α
    S[2,1] = β
    S[1,2] = -conj(β)
    S[2,2] = conj(α)
    #return S2
end

function make_submatrix(UV,i,j)
    S = zeros(ComplexF64,2,2)
    S[1,1] = UV[i,i]
    S[1,2] = UV[i,j]
    S[2,1] = UV[j,i]
    S[2,2] = UV[j,j]
    return S
end

function make_submatrix!(S,UV,i,j)
    S[1,1] = UV[i,i]
    S[1,2] = UV[i,j]
    S[2,1] = UV[j,i]
    S[2,2] = UV[j,j]
    return 
end


function make_largematrix(K,i,j,NC)
    A = zeros(ComplexF64,NC,NC)
    for n=1:NC
        A[n,n] = 1
    end
    #K = project_onto_su2(K)
    A[i,i] = K[1,1]
    A[i,j] = K[1,2] 
    A[j,i] = K[2,1]
    A[j,j] = K[2,2]  
    return A
end

function make_largematrix!(A,K,i,j,NC)
    #A = zeros(ComplexF64,NC,NC)
    A .= 0
    for n=1:NC
        A[n,n] = 1
    end
    #K = project_onto_su2(K)
    A[i,i] = K[1,1]
    A[i,j] = K[1,2] 
    A[j,i] = K[2,1]
    A[j,j] = K[2,2]  
    return A
end

const nhit = 6
const rwidth = 0.4


"""
-------------------------------------------------c
 su2-submatrix(c) in su3 matrix(x)
        su2            su3
 k=1         <-    1-2 elements
 k=2         <-    2-3 elements
 k=3         <-    1-3 elements
 k=4          ->   1-2 elements
 k=5          ->   2-3 elements
 k=6          ->   1-3 elements
-------------------------------------------------c
"""
function submat!(x,c,n,k,id)

    if k==1
        for i=1:n
            c[1,i] = real(x[1,1,i]+x[2,2,i])*0.5
            c[2,i] = imag(x[1,2,i]+x[2,1,i])*0.5
            c[3,i] = real(x[1,2,i]-x[2,1,i])*0.5
            c[4,i] = imag(x[1,1,i]-x[2,2,i])*0.5
        end
    elseif k==2
        for i=1:n
            c[1,i] = real(x[2,2,i]+x[3,3,i])*0.5
            c[2,i] = imag(x[3,2,i]+x[2,3,i])*0.5
            c[3,i] = real(x[3,2,i]-x[2,3,i])*0.5
            c[4,i] = imag(x[2,2,i]-x[3,3,i])*0.5
        end

    elseif k==3
        for i=1:n
            c[1,i] = real(x[1,1,i]+x[3,3,i])*0.5
            c[2,i] = imag(x[3,1,i]+x[1,3,i])*0.5
            c[3,i] = real(x[1,3,i]-x[3,1,i])*0.5
            c[4,i] = imag(x[1,1,i]-x[3,3,i])*0.5
        end
    elseif k==4

        for i=1:n
            #println("i = $i")
            #println(c[:,i])
            if id[i] == 1
                x[1,1,i] = c[1,i] + im*c[4,i]
                x[1,2,i] = c[3,i] + im*c[2,i]
                x[1,3,i] = 0
                x[2,1,i] = -c[3,i] + im*c[2,i]
                x[2,2,i] = c[1,i] - im*c[4,i]
                x[2,3,i] = 0
                x[3,1,i] = 0
                x[3,2,i] = 0
                x[3,3,i] = 1

            elseif id[i] == 0
                x[1,1,i] = 1
                x[1,2,i] = 0
                x[1,3,i] = 0
                x[2,1,i] = 0
                x[2,2,i] = 1
                x[2,3,i] = 0
                x[3,1,i] = 0
                x[3,2,i] = 0
                x[3,3,i] = 1
            end 
        end
    elseif k==5
        for i=1:n
            if id[i] == 1
                x[1,1,i] = 1
                x[1,2,i] = 0
                x[1,3,i] = 0
                x[2,1,i] = 0
                x[2,2,i] = c[1,i] + im*c[4,i]
                x[2,3,i] = -c[3,i] + im*c[2,i]
                x[3,1,i] = 0
                x[3,2,i] = c[3,i] + im*c[2,i]
                x[3,3,i] = c[1,i] -im*c[4,i]

            elseif id[i] == 0
                x[1,1,i] = 1
                x[1,2,i] = 0
                x[1,3,i] = 0
                x[2,1,i] = 0
                x[2,2,i] = 1
                x[2,3,i] = 0
                x[3,1,i] = 0
                x[3,2,i] = 0
                x[3,3,i] = 1
            end 
        end

    elseif k==6
        for i=1:n
            if id[i] == 1
                x[1,1,i] = c[1,i] + im*c[4,i]
                x[1,2,i] = 0
                x[1,3,i] = c[3,i] + im*c[2,i]
                x[2,1,i] = 0
                x[2,2,i] = 1
                x[2,3,i] = 0
                x[3,1,i] = -c[3,i] + im*c[2,i]
                x[3,2,i] = 0
                x[3,3,i] = c[1,i] -im*c[4,i]

            elseif id[i] == 0
                x[1,1,i] = 1
                x[1,2,i] = 0
                x[1,3,i] = 0
                x[2,1,i] = 0
                x[2,2,i] = 1
                x[2,3,i] = 0
                x[3,1,i] = 0
                x[3,2,i] = 0
                x[3,3,i] = 1
            end 
        end
    end
end

function rndprd!(ranf,n)
    rn = zeros(Float64,n)
    rndprd!(ranf,rn,n)
    return rn
end

function rndprd!(ranf,rn,n)
    for i=1:n
        rn[i] = ranf()
    end
    return rn
end

function rndprd2!(ranf,n)
    xrn = zeros(Float64,3,n)
    rndprd2!(ranf,xrn,n)
    return xrn
end

function rndprd2!(ranf,xrn,n)
    for j=1:n
        for i=1:3
            xrn[i,j] = ranf()
        end
    end
    return 
end

end