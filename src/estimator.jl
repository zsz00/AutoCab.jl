
struct Observation
    pixel::Point2f
    point::Point3f
    pose::NTuple{6, Float64}

    point_order::Int64
    pose_order::Int64

    constant::Bool   # 不变的
    in_covmap::Bool   # 

    kfid::Int64    # key frame id
    mpid::Int64    # map id
end

struct LocalBACache
    observations::Vector{Observation}
    outliers::Vector{Bool} # same length as observations

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

"""
Estimator 负责执行 ba和map过滤
"""
mutable struct Estimator
    map_manager::MapManager
    params::Params

    frame_queue::Vector{Frame}
    new_kf_available::Bool
    exit_required::Bool
end

function Estimator(map_manager::MapManager, params::Params)
    Estimator(map_manager, params, Frame[], false, false)
end

"""
启动Estimator的主程序.
在mapper.Mapper()里被调用
"""
function run!(estimator::Estimator)
    while !estimator.exit_required
        new_kf = get_new_kf!(estimator)
        local_bundle_adjustment!(estimator, new_kf)
    end
end

function add_new_kf!(estimator::Estimator, frame::Frame)
    push!(estimator.frame_queue, frame)
    estimator.new_kf_available = true
end


function get_new_kf!(estimator::Estimator)
    if isempty(estimator.frame_queue)
        estimator.new_kf_available = false
        return nothing
    end

    # TODO if more than 1 frame in queue, add them to ba anyway.
    estimator.new_kf_available = false
    kf = popfirst!(estimator.frame_queue)
    kf
end

function _get_ba_parameters(
    map_manager::MapManager, frame::Frame,
    covisibility_map::OrderedDict{Int64, Int64}, min_cov_score,
)
    # poses: kfid → (order id, θ).
    poses = Dict{Int64, Tuple{Int64, NTuple{6, Float64}}}()
    constant_poses = Set{Int64}()   # 常量pose
    # map_points: mpid → (order id, point).
    map_points = Dict{Int64, Tuple{Int64, Point3f}}()

    processed_keypoints_ids = Set{Int64}()  # 处理过的关键点

    observations = Vector{Observation}(undef, 0) # 观测值
    poses_remap = Vector{Int64}(undef, 0)   # 重映射id
    points_remap = Vector{Int64}(undef, 0)   # 重映射id
    sizehint!(observations, 1000)
    sizehint!(poses_remap, 10)
    sizehint!(points_remap, 1000)

    # 收集所有观测点 observations
    for (co_kfid, score) in covisibility_map  # 可变的map/3d点
        co_frame = get_keyframe(map_manager, co_kfid)  # 获取 可变frame

        if !(co_kfid in keys(poses))   # 如果co_kfid不在poses里
            if !(co_kfid in constant_poses)
                is_constant = score < min_cov_score || co_kfid == 0
                is_constant && (push!(constant_poses, co_kfid); continue)
            end
        end

        for kpid in get_3d_keypoints_ids(co_frame)
            kpid in processed_keypoints_ids && continue
            push!(processed_keypoints_ids, kpid)

            mp = get_mappoint(map_manager, kpid)  # 3d point
 
            mp_order_id = length(map_points) + 1   # 顺序id
            mp_position = get_position(mp)  # 3d点的坐标
            map_points[kpid] = (mp_order_id, mp_position)   # 3d dict
            push!(points_remap, kpid)

            # For each observer, add observation: px ← (mp, pose).
            for ob_kfid in get_observers(mp)   # 每个3d点对应的所有相机投影点
                ob_kfid > frame.kfid && continue

                ob_frame = get_keyframe(map_manager, ob_kfid)
                ob_pixel = get_keypoint_unpx(ob_frame, kpid)

                in_processed = ob_kfid in keys(poses)
                in_constants = ob_kfid in constant_poses
                in_covmap = ob_kfid in keys(covisibility_map)

                is_constant = ob_kfid == 0 || in_constants || !in_covmap
                !is_constant && in_covmap && (is_constant = covisibility_map[ob_kfid] < min_cov_score;)

                if in_processed
                    pose_order_id, ob_pose = poses[ob_kfid]
                else
                    ob_pose = get_cw_ba(ob_frame)  # 获取到ob_frame的 pose:r_vec,t
                    pose_order_id = length(poses) + 1
                    poses[ob_kfid] = (pose_order_id, ob_pose)

                    push!(poses_remap, ob_kfid)
                    is_constant && push!(constant_poses, ob_kfid)
                end
                
                # 每一个 投影点ob_pixel,是一个Observation对象.
                push!(observations, Observation(ob_pixel, mp_position, ob_pose,
                    mp_order_id, pose_order_id, is_constant, in_covmap, ob_kfid, kpid))
            end
        end
    end

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

function _update_ba_parameters!(map_manager::MapManager, cache::LocalBACache, current_kfid)
    points_shift = length(cache.poses_remap) * 6

    for (i, kfid) in enumerate(cache.poses_remap)
        p = (i - 1) * 6
        kf = get_keyframe(map_manager, kfid)
        set_cw_ba!(kf, @view(cache.θ[(p + 1):(p + 6)]))
    end

    for (i, mpid) in enumerate(cache.points_remap)
        mp = get_mappoint(map_manager, mpid)
        p = points_shift + (i - 1) * 3
        set_position!(mp, @view(cache.θ[(p + 1):(p + 3)]))
    end

end


function local_bundle_adjustment!(estimator::Estimator, new_frame::Frame)

    estimator.params.local_ba_on = true
    covisibility_map = get_covisible_map(new_frame)   # 可改的map/3d点
    covisibility_map[new_frame.kfid] = new_frame.nb_3d_kpts

    # Get up to 5 latest KeyFrames. 追赶最新的5个KeyFrames
    co_kfids = sort!(collect(keys(covisibility_map)); rev=true)
    co_kfids = co_kfids[1:min(5, length(co_kfids))]
    covisibility_map = OrderedDict{Int64, Int64}(kfid => covisibility_map[kfid] for kfid in co_kfids)

    cache = _get_ba_parameters(estimator.map_manager, new_frame, covisibility_map, estimator.params.min_cov_score)
    
    bundle_adjustment!(cache, new_frame.camera; show_trace=false)

    _update_ba_parameters!(estimator.map_manager, cache, new_frame.kfid)

    estimator.params.local_ba_on = false
end
