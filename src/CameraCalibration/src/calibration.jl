"""
Based on:
Wilhelm Burger: Zhang's Camera Calibration Algorithm: In-Depth Tutorial and Implementation,
Technical Report HGB16-05, University of Applied Sciences Upper Austria, School of
Informatics, Communications and Media, Dept. of Digital Media, Hagenberg, Austria, May
2016. DOI: 10.13140/RG.2.1.1166.1688, http://staff.fh-hagenberg.at/burger/
"""

"""
estimateHomography - returns the estimated homography matrix H, such that b_j = H*a_j.
'''
A and B are in the form 3xN (each column is a point)
'''
"""
function estimateHomography(a::Array{<:Real, 2}, b::Array{<:Real, 2})
    if size(a) != size(b)
        error("src and dst must have same dimensions")
    end
    n = size(a)[2] #number of points
    #normalisation matrices
    nA = getNormalisationMatrix(a)
    nB = getNormalisationMatrix(b)
    m = zeros((2 * n, 9))

    #construct matrix for solving
    for j in 1:n
        k = 2 * j
        ap = nA * a[:, j]
        bp = nB * b[:, j]
        # M_{k,*} = (x′_a, y′_a, 1, 0, 0, 0, −x′_ax′_b , −y′_ax′_b, −x′_b )
        m[k - 1, :] = [ap[1] ap[2] 1 0 0 0 -ap[1] * bp[1] -ap[2] * bp[1] -bp[1]]
        # M_{k+1,*} = (0, 0, 0, x′_a , y′_a , 1, −x′_ay′_b, −y′_ay′_b , −y′_b )
        m[k, :] = [0 0 0 ap[1] ap[2] 1 -ap[1] * bp[2] -ap[2] * bp[2] -bp[2]]
    end

    #solve
    u, d, v = svd(m, full=true)
    h = v[:, 9] #last column of v
    H = [h[1] h[2] h[3];
         h[4] h[5] h[6]
         h[7] h[8] h[9]]

    H = inv(nB) * H * nA
    return H ./ H[3, 3]
end
"""
getNormalisationMatrix - returns normalisation matrix of points x
'''

'''
"""
function getNormalisationMatrix(x::Array{<:Real, 2})
    m = mean(x[1:2, :], dims=2)
    v = var(x[1:2, :], dims=2, corrected=false, mean=m)
    #scale values (182)
    sx = sqrt(2.0 / v[1])
    sy = sqrt(2.0 / v[2])
    #construct normalisation matrix (181)
    return [sx 0 -sx*m[1]; 0 sy -sy*m[2]; 0 0 1]
end

function refineHomography(h::Array{<:Real, 2}, a::Array{<:Real, 2}, b::Array{<:Real, 2})
    n = size(a)[2]
    X = zeros((2, 2 * n))
    Y = zoeros((2, 2 * n))
    for i in 1:n
    end
end

function homographyVal(x::Array{<:Real, 2}, h::Array{<:Real, 1})
    n = size(x)[2]
    y = zeros(2 * n)
    for j in 1:n
        w = h[7] * x[1, j] + h[8] * x[2, j] + h[9]
        y[j * 2 - 1] = (h[1] * x[1, j] + h[2] * x[2, j] + h[3]) / w
        y[j * 2] = (h[4] * x[1, j] + h[5] * x[2, j] + h[6]) / w
    end
    return y
end

function homographyJac(x::Array{<:Real, 2}, h::Array{<:Real, 1})
    n = size(x)[2]
    J = zeros((2 * n, 9))
    for j in 1:n
        sx = h[1] * x[1, j] + h[2] * x[2, j] + h[3]
        sy = h[4] * x[1, j] + h[5] * x[2, j] + h[6]
        w = h[7] * x[1, j] + h[8] * x[2, j] + h[9]
        J[2j - 1, :] = [x[1, j] / w x[2, j] / w 1 / w 0 0 0 -sx * x[1, j] / (w^2) -sx *
                                                                                  x[2, j] /
                                                                                  (w^2) -sx /
                                                                                        (w^2)]
        J[2j, :] = [0 0 0 x[1, j] / w x[2, j] / w 1 / w -sy * x[1, j] / (w^2) -sy *
                                                                              x[2, j] /
                                                                              (w^2) -sy /
                                                                                    (w^2)]
    end
    return J
end

function calibrate(x::Array{<:Real, 2}, u::Array{Array{T1, 2}, 1}) where {T1 <: Real}
    hListInit = getHomographies(x, u)
    aInit = getCameraIntrinsicsB(hListInit)
    wInit = getExtrinsics(aInit, hListInit)
end

function getHomographies(x::Array{<:Real, 2}, u::Array{Array{T1, 2}, 1}) where {T1 <: Real}
    m = size(u)[1]
    hList = Array{Float64, 2}[]
    for i in 1:m
        hInit = estimateHomography(x, u[i])
        # TODO refineHomography
        push!(hList, hInit)
    end
    return hList
end

function getCameraIntrinsics(hs::Array{Array{T1, 2}, 1}) where {T1 <: Real}
    m = size(hs)[1]
    V = zeros((2 * m, 6))

    for i in 1:m
        V[2 * i - 1, :] = getIntrinsicRowVector(1, 2, hs[i])
        V[2 * i, :] = getIntrinsicRowVector(1, 1, hs[i]) -
                      getIntrinsicRowVector(2, 2, hs[i])
    end

    #solve
    u, s, v = svd(V, full=true)
    print(size(v))
    b = v[:, 6]

    w = b[1] * b[3] * b[6] - (b[2]^2) * b[6] - b[1] * (b[5]^2) + 2 * b[2] * b[4] * b[5] -
        b[3] * (b[4]^2) # (104)
    d = b[1] * b[3] - b[2]^2 # (105)
    α = sqrt(w / (d * b[1])) # (99)
    β = sqrt(w / d^2 * b[1]) # (100)
    γ = sqrt(w / (d^2 * b[1])) * b[2] # (101)
    uc = (b[2] * b[5] - b[3] * b[4]) / d
    vc = (b[2] * b[4] - b[1] * b[5]) / d
    return [α γ uc
            0 β vc
            0 0 1]
end

function getCameraIntrinsicsB(hs::Array{Array{T1, 2}, 1}) where {T1 <: Real}
    m = size(hs)[1]
    V = zeros((2 * m, 6))

    for i in 1:m
        V[2 * i - 1, :] = getIntrinsicRowVector(1, 2, hs[i])
        V[2 * i, :] = getIntrinsicRowVector(1, 1, hs[i]) .-
                      getIntrinsicRowVector(2, 2, hs[i])
    end

    #solve
    u, s, v = svd(V, full=true)
    b = v[:, 6]
    if b[1] < 0 || b[3] < 0 || b[6] < 0
        b = -b
    end
    bInit = [b[1] b[2] b[4]
             b[2] b[3] b[5]
             b[4] b[5] b[6]]
    C = cholesky(bInit)
    return inv(C.L)' * C.L[3, 3]
end

function getIntrinsicRowVector(p::Int64, q::Int64, h::Array{<:Real, 2})
    return [h[1, p] * h[1, q]
            h[1, p] * h[2, q] + h[2, p] * h[1, q]
            h[2, p] * h[2, q]
            h[3, p] * h[1, q] + h[1, p] * h[3, q]
            h[3, p] * h[2, q] + h[2, p] * h[3, q]
            h[3, p] * h[3, q]]'
end

"""
a is intrinsic matrix hs is list of homographies

"""
function getExtrinsics(a::Array{<:Real, 2}, hs::Array{Array{T1, 2}, 1}) where {T1 <: Real}
    wList = Array{Float64, 2}[]
    m = size(hs)[1]

    for i in 1:m
        push!(wList, estimateViewTransorm(a, hs[i]))
    end
    return wList
end

function estimateViewTransorm(a::Array{<:Real, 2}, h::Array{<:Real, 2})
    κ = 1 / norm(inv(a) * h[:, 1])
    r0 = κ * inv(a) * h[:, 1]
    r1 = κ * inv(a) * h[:, 2]
    r2 = cross(r0, r1)
    R = zeros((3, 3))
    R[:, 1] = r0
    R[:, 2] = r1
    R[:, 3] = r2
    # make true rotation matrix
    u, s, v = svd(R, full=true)
    R = v * u'
    t = κ * inv(a) * h[:, 3]
    return hcat(R, t)
end
