using StaticArrays
using LinearAlgebra
using Images
using JSON3
# using PyCall
using PythonCall
using RecoverPose
using Rotations
using Manifolds

const Point2 = SVector{2}
const Point2f = SVector{2, Float32}
const Point3f = SVector{3, Float32}
const SE3 = SpecialEuclidean(3)

include("camera.jl")
include("frame.jl")
include("estimator.jl")
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
    # return
    cam = Camera(
                1, K[1,1], K[2,2], K[1,3], K[2,3], # Intrinsics.
                distCoeffs[1], distCoeffs[2], distCoeffs[3], distCoeffs[4], # Distortion coefficients.
                2048, 2448
    )
    previous_points = Vector{Point2f}(undef, length(imgpoints[0]))
    current_points = Vector{Point2f}(undef, length(imgpoints[0]))
    previous_pd = Vector{Point2f}(undef, length(imgpoints[0]))  # 归一化的, predivided by K^-1 
    current_pd = Vector{Point2f}(undef, length(imgpoints[0]))

    @inbounds for i in range(1,length=length(imgpoints[0]))
        # Convert points to `(x, y)` format as expected by five points.
        previous_points[i] = Point2f(previous_pts[i,:,:]) 
        current_points[i] = Point2f(current_pts[i,:,:])
        previous_pd[i] = Point2f(backproject(cam, previous_points[i])[[1, 2]]) 
        current_pd[i] = Point2f(backproject(cam, current_points[i])[[1, 2]])
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

    r_mat_rel_gt = Array([[9.99832519e-01, -1.82957663e-02, -4.45082354e-04],
                          [1.82940228e-02, 9.99825994e-01, -3.64845807e-03],
                          [5.11756243e-04, 3.63970467e-03, 9.99993245e-01]])
    R = np.array(pyrowlist(R))   # julia 转到   python np.array()
    r_mat_rel_gt = np.array(pyrowlist(r_mat_rel_gt))
    info, _ = camera.get_euler_angle(R)
    degrees = camera.get_r_error(R, r_mat_rel_gt)
    println(info)
    println("degrees:", degrees)
end

function backproject(c::Camera, point::Point2f)
    # (x,y) --> (x',y',1)
    Point3f((point[1] - c.cx) / c.fx, (point[2] - c.cy) / c.fy, 1.0)
end


function ba_1()
    # local_ba
    # map_manager是最顶层的对象了

    cache = _get_ba_parameters(map_manager, covisibility_map, min_cov_score)
    
    bundle_adjustment!(cache, camera; show_trace=false)
 
end



camera_1()




#=
export JULIA_PYTHONCALL_EXE=/home/zhangyong/miniconda3/bin/python
julia --project=/home/zhangyong/codes/AutoCab.jl/Project.toml /home/zhangyong/codes/AutoCab.jl/src/test_2.jl


=#
