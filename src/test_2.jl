using StaticArrays
using LinearAlgebra
using Images
using JSON3, JLD2
# using PyCall
using PythonCall
using RecoverPose
using Rotations
using Manifolds
using OrderedCollections: OrderedSet, OrderedDict
using LeastSquaresOptim
using SparseArrays
using SparseDiffTools

const Point2 = SVector{2}
const Point2f = SVector{2, Float32}
const Point3f = SVector{3, Float32}
const SE3 = SpecialEuclidean(3)

# 转换为齐次坐标
@inline to_homogeneous(p::SVector{3, T}) where T = SVector{4, T}(p..., one(T))
@inline to_homogeneous(p::SVector{4}) = p

function to_4x4(m::SMatrix{3, 3, T, 9}) where T
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        0,       0,       0,       1)
end

include("camera.jl")
include("frame.jl")
include("estimator_2.jl")
include("bundle_adjustment.jl")

@py import sys
sys.path.insert(0, Py("/home/zhangyong/codes/AutoCab.jl/src"))
camera = pyimport("camera")
np = pyimport("numpy")


function camera_1()
    data = camera.normal_camera_calibrate()
    # camera.test()
    objpoints, imgpoints, images, cameraMatrix, distCoeffs, rvecs, tvecs, rvecs_matrixs = data
    
    println("=============================================")
    println(imgpoints[0][10])

    objpoints = pyconvert(Array{Float32, 2}, objpoints[0])
    previous_pts = pyconvert(Array{Float32, 3}, imgpoints[0])
    current_pts = pyconvert(Array{Float32, 3}, imgpoints[1])
    K = pyconvert(Array{Float64, 2}, cameraMatrix)
    K_1 = Array{Float64}([1 0 0; 0  1  0; 0  0  1])
    distCoeffs = pyconvert(Array{Float64, 2}, distCoeffs)[1,:]
    println(size(objpoints), size(previous_pts))
    println(objpoints[11,:,:], previous_pts[11,:,:])
    println(distCoeffs)
    cameras = Vector{Camera}()
    cam_1 = Camera(
                1, K[1,1], K[2,2], K[1,3], K[2,3], # Intrinsics.
                distCoeffs[1], distCoeffs[2], distCoeffs[3], distCoeffs[4], # Distortion coefficients.
                2048, 2448
    )
    cam_2 = Camera(
                2, K[1,1], K[2,2], K[1,3], K[2,3], # Intrinsics.
                distCoeffs[1], distCoeffs[2], distCoeffs[3], distCoeffs[4], # Distortion coefficients.
                2048, 2448
    )
    cameras = [cam_1, cam_2]

    previous_points = Vector{Point2f}(undef, length(imgpoints[0]))
    current_points = Vector{Point2f}(undef, length(imgpoints[0]))
    previous_pd = Vector{Point2f}(undef, length(imgpoints[0]))  # 归一化的, predivided by K^-1 
    current_pd = Vector{Point2f}(undef, length(imgpoints[0]))

    @inbounds for i in range(1,length=length(imgpoints[0]))
        # Convert points to `(x, y)` format as expected by five points.
        previous_points[i] = Point2f(previous_pts[i,:,:]) 
        current_points[i] = Point2f(current_pts[i,:,:])
        previous_pd[i] = Point2f(backproject(cam_1, previous_points[i])[[1, 2]]) 
        current_pd[i] = Point2f(backproject(cam_1, current_points[i])[[1, 2]])
    end
    println("==========: ", previous_points[11])
    println("==========: ", previous_pd[11])

    n_inliers, (E, P, inliers, best_repr_error) = five_point_ransac(
        previous_points, current_points, K_1, K_1, GEEV4x4Cache(); max_repr_error=0.001, iterations=8500, confidence=0.999)

    focal_sum = 2  # 5243.8974
    # n_inliers, (E, inliers, best_repr_error) = essential_ransac(previous_points, current_points, focal_sum; threshold = 1.0)
    # best_n_inliers, P_res, best_inliers, best_error = recover_pose(E, pixels1, pixels2, K1, K2; threshold = 1.0)
    println("best_repr_error: ", best_repr_error)
    # 为什么求出的E,P, n_inliers 是变动的? 因为ransec是C(n,5),但是iterations有限,E就不是全局最优解. 

    println("size: ", length(inliers), " n_inliers: ", n_inliers)
    R, t = P[1:3, 1:3], P[1:3, 4]
    println("E: ", E)
    println("R: ", R)
    println("t: ", t)
    
    cameras[2].Ti0 = vcat(P, [0 0 0 1.0])  # 0->i
    
    r_mat_rel_gt = Array([[9.99832519e-01, -1.82957663e-02, -4.45082354e-04],
                          [1.82940228e-02, 9.99825994e-01, -3.64845807e-03],
                          [5.11756243e-04, 3.63970467e-03, 9.99993245e-01]])
    R = np.array(pyrowlist(R))   # julia 转到 python np.array()
    r_mat_rel_gt = np.array(pyrowlist(r_mat_rel_gt))
    info, _ = camera.get_euler_angle(R)
    degrees = camera.get_r_error(R, r_mat_rel_gt)
    println(info)
    println("degrees:", degrees)
    # save("/home/zhangyong/codes/AutoCab.jl/src/cameras_1.jld2", "aaa")
    # jldsave("/home/zhangyong/codes/AutoCab.jl/src/cameras_1.jld2"; previous_points, current_points, cameras)
    @save "/home/zhangyong/codes/AutoCab.jl/src/cameras_2.jld2" previous_points current_points cameras
    return previous_points, current_points, cameras
end

function backproject(c::Camera, point::Point2f)
    # (x,y) --> (x',y',1)
    Point3f((point[1] - c.cx) / c.fx, (point[2] - c.cy) / c.fy, 1.0)
end

function get_observations(previous_points, current_points, cameras)
    # previous_points, current_points 是成对的匹配点
    K = to_4x4(cameras[1].K)
    # P1 - previous Keyframe, P2 - this `frame`.
    P1 = K * SMatrix{4, 4, Float64, 16}(I)
    P2 = K * cameras[2].Ti0
    set_cw!(cameras[2], cameras[2].Ti0)

    observations = []
    mp_order_id = 0
    for (kpup, obup) in zip(previous_points, current_points)
        println(kpup, obup)
        left_point = triangulate(obup, kpup, P1, P2)  # 2d->3d点, 齐次的. 相机坐标的
        left_point *= 1.0 / left_point[4]
        println(left_point)
        wpt = project_camera_to_world(cameras[2], left_point)[1:3]  # 相机坐标系到世界坐标系
        mp_position = wpt  # 3d point
        println(mp_position)

        ob_pixel = obup
        ob_pose = get_cw_ba(cameras[2])   # R,t
        mp_order_id += 1
        # 观察点
        observation = Observation(ob_pixel, mp_position, ob_pose, mp_order_id, 1, false, false)
        push!(observations, observation)
    end
    
    return observations
end

function ba_1(observations, cameras)
    # local_ba
    println("ba_1()")
    cache = _get_ba_parameters(observations)
    println(cache.θ[1:6])
    bundle_adjustment!(cache, cameras[2])
    println(cache.θ[1:6])

end

function test_2()
    println("test_2()")
    # previous_points, current_points, cameras = camera_1()
    @load "/home/zhangyong/codes/AutoCab.jl/src/cameras_2.jld2" previous_points current_points cameras

    observations = get_observations(previous_points, current_points, cameras)

    ba_1(observations, cameras)
    
end


# camera_1()
test_2()


#=
10.9.1.8
export JULIA_PYTHONCALL_EXE=/home/zhangyong/miniconda3/bin/python
julia --project=/home/zhangyong/codes/AutoCab.jl/Project.toml /home/zhangyong/codes/AutoCab.jl/src/test_2.jl


=#
