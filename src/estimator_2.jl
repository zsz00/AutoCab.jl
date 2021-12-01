
struct Observation
    pixel::Point2f
    point::Point3f   # 3d, 世界坐标系 
    pose::NTuple{6, Float64}   # NTuple,长度为6个,类型都是Float64的Tuple
    # camera_id::Int64
    point_order::Int64
    pose_order::Int64    # camera_id

    constant::Bool   # 不变的
    in_covmap::Bool   # 是否在不变3d点里
    kfid::Int64    # key frame id
    mpid::Int64    # map id
end

struct LocalBACache
    observations::Vector{Observation}
    outliers::Vector{Bool} # same length as observations. 异常值标志

    θ::Vector{Float64}
    θconst::Vector{Bool}
    pixels::Matrix{Float64}

    poses_ids::Vector{Int64}
    points_ids::Vector{Int64}
    poses_remap::Vector{Int64} # order id → kfid
    points_remap::Vector{Int64} # order id → mpid
end

function LocalBACache(
    observations, θ, θconst, pixels,
    poses_ids, points_ids, poses_remap, points_remap,
)
    outliers = Vector{Bool}(undef, length(observations))
    LocalBACache(
        observations, outliers, θ, θconst, pixels,
        poses_ids, points_ids, poses_remap, points_remap)
end


function _get_ba_parameters(observations)
    # poses: kfid → (order id, θ).  所有的poses
    poses = Dict{Int64, Tuple{Int64, NTuple{6, Float64}}}()
    constant_poses = Set{Int64}()   # 常量pose
    # map_points: mpid → (order id, point).
    map_points = Dict{Int64, Tuple{Int64, Point3f}}()
    processed_keypoints_ids = Set{Int64}()  # 处理过的3d关键点

    # observations = Vector{Observation}(undef, 0) # 观测值
    poses_remap = Vector{Int64}(undef, 0)   # 重映射id
    points_remap = Vector{Int64}(undef, 0)   # 重映射id
    sizehint!(observations, 1000)
    sizehint!(poses_remap, 10)
    sizehint!(points_remap, 1000)

    # observations = get_observations(imgpoints, R, t)

    n_observations = length(observations)
    n_poses, n_points = length(poses), length(map_points)
    point_shift = n_poses * 6

    θ = Vector{Float64}(undef, point_shift + n_points * 3)
    θconst = Vector{Bool}(undef, n_poses)
    poses_ids = Vector{Int64}(undef, n_observations)
    points_ids = Vector{Int64}(undef, n_observations)
    pixels = Matrix{Float64}(undef, 2, n_observations)

    processed_poses = fill(false, n_poses)
    processed_points = fill(false, n_points)

    for (oi, observation) in enumerate(observations)
        pixels[:, oi] .= observation.pixel  # 像素坐标
        poses_ids[oi] = observation.pose_order  # pose 
        points_ids[oi] = observation.point_order

        if !processed_poses[observation.pose_order]
            processed_poses[observation.pose_order] = true
            p = (observation.pose_order - 1) * 6

            θ[(p + 1):(p + 6)] .= observation.pose
            θconst[observation.pose_order] = observation.constant
        end
        if !processed_points[observation.point_order]
            processed_points[observation.point_order] = true
            p = point_shift + (observation.point_order - 1) * 3
            θ[(p + 1):(p + 3)] .= observation.point
        end
    end

    local_ba_cache = LocalBACache(observations, θ, θconst, pixels, poses_ids, points_ids, poses_remap, points_remap)
    return local_ba_cache
end


