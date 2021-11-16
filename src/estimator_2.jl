
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

function _get_ba_parameters()
    # poses: kfid → (order id, θ).  所有的poses
    poses = Dict{Int64, Tuple{Int64, NTuple{6, Float64}}}()
    constant_poses = Set{Int64}()   # 常量pose
    # map_points: mpid → (order id, point).
    map_points = Dict{Int64, Tuple{Int64, Point3f}}()
    processed_keypoints_ids = Set{Int64}()  # 处理过的3d关键点

    observations = Vector{Observation}(undef, 0) # 观测值
    poses_remap = Vector{Int64}(undef, 0)   # 重映射id
    points_remap = Vector{Int64}(undef, 0)   # 重映射id
    sizehint!(observations, 1000)
    sizehint!(poses_remap, 10)
    sizehint!(points_remap, 1000)

    observations = get_observations(imgpoints, R, t)

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


function local_bundle_adjustment!(map_manager::MapManager, new_frame::Frame)

    covisibility_map = get_covisible_map(new_frame)   # 可改的map/3d点
    covisibility_map[new_frame.kfid] = new_frame.nb_3d_kpts  # {kfid:3d点的数量}

    # Get up to 5 latest KeyFrames. 追赶最新的5个KeyFrames
    co_kfids = sort!(collect(keys(covisibility_map)); rev=true)
    co_kfids = co_kfids[1:min(5, length(co_kfids))]
    covisibility_map = OrderedDict{Int64, Int64}(kfid => covisibility_map[kfid] for kfid in co_kfids)

    cache = _get_ba_parameters()
    
    bundle_adjustment!(cache, new_frame.camera; show_trace=false)

    # _update_ba_parameters!(map_manager, cache, new_frame.kfid)
end


function get_observations(imgpoints, cameras)
    # imgpoints是 成对的匹配点
    K = to_4x4(camera.K)
    # P1 - previous Keyframe, P2 - this `frame`.
    P1 = K * SMatrix{4, 4, Float64, 16}(I)
    P2 = K * SMatrix{4, 4, Float64, 16}(I)

    observations = []
    for 2d_pint in imgpoints
        
        left_point = triangulate(obup[[2, 1]], kpup[[2, 1]], P1, P2)  # 2d->3d点, 齐次的. 相机坐标的
        left_point *= 1.0 / left_point[4]
        wpt = project_camera_to_world(cameras[2], left_point)[1:3]  # 相机坐标系到世界坐标系
        mp_position = wpt

        # observation = Observation(ob_pixel, mp_position, ob_pose, mp_order_id, 
        #                           pose_order_id, is_constant, in_covmap, ob_kfid, kpid)
        # push!(observations, observation)

    end
    
    return observations
end

