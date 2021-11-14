module AutoCab
export CabManager, add_image!, add_stereo_image!, get_queue_size
export Params, Camera, run!, to_cartesian, reset!
export Visualizer, ReplaySaver
export set_frame_wc!, process_frame_wc!, set_image!, set_position!

using BSON: @save, @load
using OrderedCollections: OrderedSet, OrderedDict
# using GLMakie
using Interpolations
using Images
using ImageDraw
using ImageFeatures
using LeastSquaresOptim
using LinearAlgebra
using Manifolds
using Random
using RecoverPose
using Rotations
using StaticArrays
using SparseArrays
using SparseDiffTools

const Point2 = SVector{2}
const Point2i = SVector{2, Int64}
const Point2f = SVector{2, Float64}
const Point3f = SVector{3, Float64}
const Point3f0 = SVector{3, Float32}

const SE3 = SpecialEuclidean(3)

@inline convert(x::Point2f)::Point2i = x .|> round .|> Int64
@inline convert(x::Vector{Point2f}) = Point2i[xi .|> round .|> Int64 for xi in x]
# 转换为齐次坐标
@inline to_homogeneous(p::SVector{3, T}) where T = SVector{4, T}(p..., one(T))
@inline to_homogeneous(p::SVector{4}) = p

@inline to_cartesian(x) = CartesianIndex(convert(x)...)
@inline function to_cartesian(x::Point2, cell_size::Int64)
    x = convert(x) .÷ cell_size .+ 1
    CartesianIndex(x...)
end

function to_4x4(m::SMatrix{3, 3, T, 9}) where T
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        0,       0,       0,       1)
end
function to_4x4(m::SMatrix{3, 4, T, 12}) where T
    SMatrix{4, 4, T}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        m[1, 4], m[2, 4], m[3, 4], 1)
end
function to_4x4(m, t)
    SMatrix{4, 4, Float64}(
        m[1, 1], m[2, 1], m[3, 1], 0,
        m[1, 2], m[2, 2], m[3, 2], 0,
        m[1, 3], m[2, 3], m[3, 3], 0,
        t[1],    t[2],    t[3],    1)
end

include("camera.jl")
# include("extractor.jl")
# include("tracker.jl")
include("params.jl")
include("frame.jl")
# include("motion_model.jl")
include("map_point.jl")
include("map_manager.jl")
include("front_end.jl")
include("estimator.jl")
include("mapper.jl")
# include("io/visualizer.jl")
# include("io/saver.jl")
include("bundle_adjustment.jl")


mutable struct CabManager
    params::Params

    image_queue::Vector{Matrix{Gray{Float64}}}
    right_image_queue::Vector{Matrix{Gray{Float64}}}
    time_queue::Vector{Float64}

    current_frame::Frame
    frame_id::Int64

    front_end::FrontEnd
    map_manager::MapManager
    mapper::Mapper
    extractor::Extractor

    visualizer::Union{Nothing, Visualizer, ReplaySaver}

    exit_required::Bool

    mapper_thread
end

function CabManager(params, camera; right_camera = nothing, visualizer = nothing,)

    avoidance_radius = max(5, params.max_distance ÷ 2)
    image_resolution = (camera.height, camera.width)  # 图像分辨率
    grid_resolution = ceil.(Int64, image_resolution ./ params.max_distance)

    image_queue = Matrix{Gray{Float64}}[]
    right_image_queue = Matrix{Gray{Float64}}[]
    time_queue = Float64[]

    frame = Frame(;camera, right_camera, cell_size=params.max_distance)  # 定义frame
    extractor = Extractor(params.max_nb_keypoints, avoidance_radius,
                          grid_resolution, params.max_distance)  # 提取特征点
    map_manager = MapManager(params, frame, extractor)   # 数据管理. frame,point,pose
    front_end = FrontEnd(params, frame, map_manager)  # 计算初始pose
    
    mapper = Mapper(params, map_manager, frame)   # map匹配和优化
    mapper_thread = Threads.@spawn run!(mapper)

    CabManager(params, image_queue, right_image_queue, time_queue, frame, frame.id,
               front_end, map_manager, mapper, extractor, visualizer, false, mapper_thread)
end

function run!(sm::CabManager)
    image::Union{Nothing, Matrix{Gray{Float64}}} = nothing

    while !sm.exit_required
        image, time = get_image!(sm)

        sm.frame_id += 1
        sm.current_frame.id = sm.frame_id
        sm.current_frame.time = time
        # 向mapper 里加入keyframe
        add_new_kf!(sm.mapper, KeyFrame(sm.current_frame.kfid, nothing))
        sleep(1e-2)
    end
end

function add_image!(sm::CabManager, image, time)
    push!(sm.image_queue, image)
    push!(sm.time_queue, time)
end

function get_image!(sm::CabManager)
    isempty(sm.image_queue) && return nothing, nothing
    image = popfirst!(sm.image_queue)
    time = popfirst!(sm.time_queue)
    image, time
end

function get_queue_size(sm::CabManager)
    length(sm.image_queue)
end

function reset!(sm::CabManager)
    @warn "[Cab Manager] Reset required."
    sm.params |> reset!
    sm.current_frame |> reset!
    sm.front_end |> reset!
    sm.map_manager |> reset!
    @warn "[Cab Manager] Reset applied."
end

function draw_keypoints!(
    image::Matrix{T}, frame::Frame; right::Bool = false,
) where T <: RGB
    radius = 2
    for kp in values(frame.keypoints)
        right && !kp.is_stereo && continue

        pixel = (right && kp.is_stereo) ? kp.right_pixel : kp.pixel
        in_image(frame.camera, pixel) || continue

        color = kp.is_3d ? T(0, 0, 1) : T(0, 1, 0)
        kp.is_retracked && (color = T(1, 0, 0);)
        draw!(image, CirclePointRadius(to_cartesian(pixel), radius), color)
    end
    image
end

end
