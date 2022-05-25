ENV["JULIA_PYTHONCALL_EXE"]="/home/zhangyong/miniconda3/bin/python"
using StaticArrays
using LinearAlgebra
using Images
using JSON3, JLD2
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
np = pyimport("numpy")
pd = pyimport("pandas")


function camera_2()
    # 获取相机初始参数,并转换格式.
    dir_1 = "/home/zhangyong/codes/auto_calibrat/images"
    out_path = joinpath(dir_1, "camera_and_view_pair_data_8_1.pkl") 
    camera_param_dict, camera_pairs, view_pair_data = pd.read_pickle(out_path)
    # @show camera_param_dict
    camera_param_dict = pyconvert(Dict, camera_param_dict)
    cameras = Dict{String,Camera}()
    cnt = 1
    for (id, cam) in camera_param_dict
        # @show cam
        cam = pyconvert(Dict, cam)
        K = pyconvert(Array{Float64, 2}, cam["camera_matrix"])
        distCoeffs = pyconvert(Array{Float64, 2}, cam["dist_coeffs"])[1,:]

        cam_1 = Camera(
                    cnt, id, K[1,1], K[2,2], K[1,3], K[2,3], # Intrinsics.
                    distCoeffs[1], distCoeffs[2], distCoeffs[3], distCoeffs[4], # Distortion coefficients.
                    1080, 1920
        )
        # push!(cameras, cam_1)
        cameras[id] = cam_1
        cnt += 1
    end

    view_pair_data = pyconvert(Dict, view_pair_data)

    return cameras, camera_pairs, view_pair_data
end

function backproject(c::Camera, point::Point2f)
    # (x,y) --> (x',y',1)
    Point3f((point[1] - c.cx) / c.fx, (point[2] - c.cy) / c.fy, 1.0)
end

function get_observations_2(cameras, view_pair_data)
    # camera_param_dict, camera_pairs, view_pair_data   previous_points, current_points, cameras
    # previous_points, current_points 是成对的匹配点
    K_1 = Array{Float64}([1 0 0; 0  1  0; 0  0  1])
    P1 = to_4x4(SMatrix{3,3,Float64,9}(K_1)) * SMatrix{4, 4, Float64, 16}(I)

    observations = []
    mp_order_id = 0
    first_cam = "河西浦口道支行大厅环境6_12"  # 河西浦口道支行大厅环境5_11 河西浦口道支行大厅环境6_12
    for (one_view_pair, view_data) in view_pair_data
        @show one_view_pair
        if ! (one_view_pair in [(first_cam, "河西浦口道支行大厅环境7_13"),(first_cam, "河西浦口道支行大厅环境8_14")])
            continue
        end
        @py one_pair_ud_crpd = view_pair_data[one_view_pair]["view_pair_ud"]
        @py one_pair_crpd = view_pair_data[one_view_pair]["view_pair_org"]

        one_pair_ud_crpd = pyconvert(Array{Float32, 3}, one_pair_ud_crpd)
        one_pair_crpd = pyconvert(Array{Float32, 3}, one_pair_crpd)

        ud_ref_view_pts_1, ud_other_view_pts_1 = one_pair_ud_crpd[:, :, 1], one_pair_ud_crpd[:, :, 2]
        ref_view_pts_1, other_view_pts_1 = one_pair_crpd[:, :, 1], one_pair_crpd[:, :, 2]
        ref_cam_id, other_cam_id = one_view_pair
        println(size(ud_ref_view_pts_1), size(ud_other_view_pts_1))

        ud_ref_view_pts = Vector{Point2f}(undef, size(ud_ref_view_pts_1,1))   # # 归一化的, pre divided by K^-1
        ud_other_view_pts = Vector{Point2f}(undef, size(ud_other_view_pts_1, 1))
        @inbounds for i in range(1,length=size(ud_ref_view_pts,1))
            # Convert points to `(x, y)` format as expected by five points.
            ud_ref_view_pts[i] = Point2f(ud_ref_view_pts_1[i,:]) 
            ud_other_view_pts[i] = Point2f(ud_other_view_pts_1[i,:])
        end
        # 求基本外参
        n_inliers, (E, P, inliers, best_repr_error) = RecoverPose.five_point_ransac(
            ud_ref_view_pts, ud_other_view_pts, K_1, K_1, GEEV4x4Cache(); 
            max_repr_error=0.03, iterations=2000, confidence=0.999)
        println("size: $(length(inliers)), n_inliers:$(n_inliers), best_repr_error: $best_repr_error, $(size(P))")
        R, t = P[1:3, 1:3], P[1:3, 4]  # P:(3,4)
        println("R: $R, \nt: $t")
        cameras[other_cam_id].Ti0 = vcat(P, [0 0 0 1.0])  # 0->i
        # K = to_4x4(cameras[other_cam_id].K)
        K = to_4x4(SMatrix{3,3,Float64,9}(K_1))
        # P1 - previous Keyframe, P2 - this `frame`.
        P2 = K * cameras[other_cam_id].Ti0
        set_cw!(cameras[other_cam_id], cameras[other_cam_id].Ti0)

        # continue
        
        for (kpup, obup) in zip(ud_ref_view_pts, ud_other_view_pts)
            # println(kpup, obup)
            left_point = triangulate(obup, kpup, P1, P2)  # 2d->3d点, 齐次的,4dim. 相机坐标的
            left_point *= 1.0 / left_point[4]
            # println(left_point)
            wpt = project_camera_to_world(cameras[other_cam_id], left_point)[1:3]  # 相机坐标系到世界坐标系
            mp_position = wpt  # 3d point, 世界坐标系
            @show wpt
            ob_pixel = obup
            ob_pose = get_cw_ba(cameras[other_cam_id])   # R,t
            mp_order_id += 1
            # 观察点, 从某个pose看到的2d点
            observation = Observation(ob_pixel, mp_position, ob_pose, mp_order_id, 1, false, false)
            push!(observations, observation)
        end
    end
    return observations
end


function ba_1(observations, cameras)
    # local_ba
    println("ba_1()")
    cache = _get_ba_parameters(observations)  # 获取ba相关的可训练参数
    println(cache.θ[1:6])   
    R = RotZYX(cache.θ[1:3]...)  # r_mat 3*3
    R = np.array(pyrowlist(R))   # julia 转到 python np.array()
    info, angle  = camera_py.get_euler_angle(R)
    t = normalize(cache.θ[4:6])
    println(info, ", t:", t)

    bundle_adjustment!(cache, cameras[2]; iterations=10, show_trace=true, repr_ϵ=1.0)

    println(cache.θ[1:6])   #   R,t 
    R = RotZYX(cache.θ[1:3]...)
    R = np.array(pyrowlist(R))   # julia 转到 python np.array()
    info, angle  = camera_py.get_euler_angle(R)
    t = normalize(cache.θ[4:6])
    println(info, ", t:", t)
end

function test_2()
    println("test_2()")
    cameras, camera_pairs, view_pair_data = camera_2()
    println(length(cameras))
    
    # @load "/home/zhangyong/codes/AutoCab.jl/src/cameras_4.jld2" previous_points current_points cameras
    # println("================================ ba ======================================")
    observations = get_observations_2(cameras, view_pair_data)
    ba_1(observations, cameras)
    
end


test_2()


#=
10.9.1.8
base test_2.jl, test_3.jl   2022.5.22
主要用了 estimator_2.jl,bundle_adjustment.jl

export JULIA_PYTHONCALL_EXE=/home/zhangyong/miniconda3/bin/python
julia --project=/home/zhangyong/codes/AutoCab.jl/Project.toml /home/zhangyong/codes/AutoCab.jl/src/test_4.jl

尝试多相机同时优化

怎么生成 cache
主要是 wpt/3d point 怎么算数量? 多视图同时看到,要作为一个点吗? 是

怎么走通?
在局部好像做不了ba,因为wpt对不齐. 
在全局做ba, wpt是绝对的, 可以对齐. 直接优化绝对外参.

=#
