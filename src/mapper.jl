struct KeyFrame
    id::Int64
    right_image::Union{Nothing, Matrix{Gray{Float64}}}
end

mutable struct Mapper
    params::Params
    map_manager::MapManager
    estimator::Estimator

    current_frame::Frame
    keyframe_queue::Vector{KeyFrame}

    exit_required::Bool
    new_kf_available::Bool

    estimator_thread
end

function Mapper(params::Params, map_manager::MapManager, frame::Frame)
    estimator = Estimator(map_manager, params)
    estimator_thread = run!(estimator)  # 进行相机pose的估计,local_ba  ************

    Mapper(
        params, map_manager, estimator,
        frame, KeyFrame[], false, false, estimator_thread)
end

function run!(mapper::Mapper)
    while !mapper.exit_required
        succ, kf = get_new_kf!(mapper)

        new_keyframe = get_keyframe(mapper.map_manager, kf.id)

        if new_keyframe.nb_2d_kpts > 0 && new_keyframe.kfid > 0
            # 三角测量法得到 3D point. 
            triangulate_temporal!(
                mapper.map_manager, new_keyframe,
                mapper.params.max_reprojection_error)
        end

        update_frame_covisibility!(mapper.map_manager, new_keyframe)

        if mapper.params.do_local_matching && kf.id > 0
            match_local_map!(mapper, new_keyframe)
        end

        add_new_kf!(mapper.estimator, new_keyframe)  # 向队列中加入数据 ******
    end
    mapper.estimator.exit_required = true
    @debug "[MP] Exit required."
    wait(mapper.estimator_thread)
end


# 用这个
function triangulate_temporal!(map_manager::MapManager, frame::Frame, max_error)
    keypoints = get_2d_keypoints(frame) # 从 frame获取 2d关键点

    K = to_4x4(frame.camera.K)
    # P1 - previous Keyframe, P2 - this `frame`.
    P1 = K * SMatrix{4, 4, Float64, 16}(I)
    P2 = K * SMatrix{4, 4, Float64, 16}(I)

    good = 0
    rel_kfid = -1
    # frame -> observer key frame.
    rel_pose::SMatrix{4, 4, Float64, 16} = SMatrix{4, 4, Float64, 16}(I)
    rel_pose_inv::SMatrix{4, 4, Float64, 16} = SMatrix{4, 4, Float64, 16}(I)

    for kp in keypoints
        @assert !kp.is_3d

        map_point = get_mappoint(map_manager, kp.id)  # 获取地图3D点
        map_point.is_3d && continue

        # Get first KeyFrame id from the set of mappoint observers.
        observers = get_observers(map_point)
        length(observers) < 2 && continue
        kfid = observers[1]
        observer_kf = get_keyframe(map_manager, kfid)

        # Compute relative motion between new KF & observer KF.
        # Don't recompute if the frame's ids don't change.
        if rel_kfid != kfid
            rel_pose = observer_kf.cw * frame.wc   # frame的相对pose
            rel_pose_inv = inv(SE3, rel_pose)
            rel_kfid = kfid
            P2 = K * rel_pose_inv
        end

        observer_kp = get_keypoint(observer_kf, kp.id)
        observer_kp ≡ nothing && continue
        obup = observer_kp.undistorted_pixel  # 去畸变,像素
        kpup = kp.undistorted_pixel

        # 视差, obup-重投影up
        parallax = norm(obup .- project(frame.camera, rel_pose[1:3, 1:3] * kp.position))  # 3d->2d

        left_point = triangulate(obup[[2, 1]], kpup[[2, 1]], P1, P2)  # 2d->3d点, 齐次的
        left_point *= 1.0 / left_point[4]
        left_point[3] < 0.1 && parallax > 20.0 && continue

        right_point = rel_pose_inv * left_point
        right_point[3] < 0.1 && parallax > 20.0 && continue

        lrepr = norm(project(frame.camera, left_point[1:3]) .- obup)
        lrepr > max_error && parallax > 20.0 && continue

        rrepr = norm(project(frame.camera, right_point[1:3]) .- kpup)
        rrepr > max_error && parallax > 20.0 && continue

        wpt = project_camera_to_world(observer_kf, left_point)[1:3]  # 相机坐标系到世界坐标系
        update_mappoint!(map_manager, kp.id, wpt)
        good += 1
    end
end


function match_local_map!(mapper::Mapper, frame::Frame)
    # Maximum number of MapPoints to track.
    max_nb_mappoints = 10 * mapper.params.max_nb_keypoints
    covisibility_map = get_covisible_map(frame)

    if length(frame.local_map_ids) < max_nb_mappoints
        # Get local map of the oldest covisible KeyFrame and add it
        # to the local map to `frame` to search for MapPoints.
        kfid = collect(keys(covisibility_map))[1]
        co_kf = get_keyframe(mapper.map_manager, kfid)
        while co_kf ≡ nothing && kfid > 0
            kfid -= 1
            co_kf = get_keyframe(mapper.map_manager, kfid)
        end

        co_kf ≢ nothing && union!(frame.local_map_ids, co_kf.local_map_ids)
        # TODO if still not enough, go for another round.
    end

    prev_new_map = do_local_map_matching(
        mapper, frame, frame.local_map_ids;
        max_projection_distance=mapper.params.max_projection_distance,
        max_descriptor_distance=mapper.params.max_descriptor_distance)

    isempty(prev_new_map) || merge_matches(mapper, prev_new_map)
end

function merge_matches(mapper::Mapper, prev_new_map::Dict{Int64, Int64})
    try
        for (prev_id, new_id) in prev_new_map
            merge_mappoints(mapper.map_manager, prev_id, new_id);
        end
    catch e
        showerror(stdout, e); println()
        display(stacktrace(catch_backtrace())); println()
    end
end

"""
给定一个frame及其关键点id的局部map(三角化),将各个3D点投射到frame上,找到周围的关键点(三角化?),用投影匹配周围的关键点.
最佳匹配者是被替换的新候选人.
"""
function do_local_map_matching(
    mapper::Mapper, frame::Frame, local_map::Set{Int64};
    max_projection_distance, max_descriptor_distance,
)
    prev_new_map = Dict{Int64, Int64}()
    isempty(local_map) && return prev_new_map

    # Maximum field of view. 最大视场角
    vfov = 0.5 * frame.camera.height / frame.camera.fy
    hfov = 0.5 * frame.camera.width / frame.camera.fx
    max_rad_fov = vfov > hfov ? atan(vfov) : atan(hfov)
    view_threshold = cos(max_rad_fov)

    # Define max distance from projection.
    frame.nb_3d_kpts < 30 && (max_projection_distance *= 2.0;)
    # matched kpid → [(local map kpid, distance)] TODO
    matches = Dict{Int64, Vector{Tuple{Int64, Float64}}}()

    # Go through all MapPoints from the local map in `frame`.
    for kpid in local_map
        is_observing_kp(frame, kpid) && continue
        mp = get_mappoint(mapper.map_manager, kpid)   # 获取3d点 
        mp ≡ nothing && continue
        (!mp.is_3d || isempty(mp.descriptor)) && continue

        # Project MapPoint into KeyFrame's image plane.
        position = get_position(mp)
        camera_position = project_world_to_camera(frame, position)[1:3]
        camera_position[3] < 0.1 && continue

        view_angle = camera_position[3] / norm(camera_position)
        abs(view_angle) < view_threshold && continue

        projection = project_undistort(frame.camera, camera_position)
        in_image(frame.camera, projection) || continue

        surrounding_keypoints = get_surrounding_keypoints(frame, projection)

        # Find best match for the `mp` among `surrounding_keypoints`.
        best_id, best_distance = find_best_match(
            mapper.map_manager, frame, mp, projection, surrounding_keypoints;
            max_projection_distance, max_descriptor_distance)
        best_id == -1 && continue

        match = (kpid, best_distance)
        if best_id in keys(matches)
            push!(matches[best_id], match)
        else
            matches[best_id] = Tuple{Int64, Float64}[match]
        end
    end

    for (kpid, match) in matches
        best_distance = 1e6
        best_id = -1

        for (local_kpid, distance) in match
            if distance ≤ best_distance
                best_distance = distance
                best_id = local_kpid
            end
            best_id != -1 && (prev_new_map[kpid] = best_id;)
        end
    end
    prev_new_map
end


function find_best_match(
    map_manager::MapManager, frame::Frame, target_mp::MapPoint,
    projection, surrounding_keypoints;
    max_projection_distance, max_descriptor_distance,
)
    target_mp_observers = get_observers(target_mp)
    target_mp_position = get_position(target_mp)

    # TODO parametrize descriptor size.
    min_distance = 256.0 * max_descriptor_distance
    best_distance, second_distance = min_distance, min_distance
    best_id, second_id = -1, -1

    for kp in surrounding_keypoints
        kp.id < 0 && continue
        distance = norm(projection .- kp.pixel)
        distance > max_projection_distance && continue

        mp = get_mappoint(map_manager, kp.id)
        if mp ≡ nothing
            remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
            continue
        end
        isempty(mp.descriptor) && continue

        # Check that `kp` and `target_mp` are indeed candidates for matching.
        # They should have no overlap in their observers.
        mp_observers = get_observers(mp)
        isempty(intersect(target_mp_observers, mp_observers)) || continue

        avg_projection = 0.0
        n_projections = 0

        # Compute average projection distance for the `target_mp` projected
        # into each of the `mp` observers KeyFrame.
        for observer_kfid in mp_observers
            observer_kf = get_keyframe(map_manager, observer_kfid)
            observer_kf ≡ nothing && (remove_mappoint_obs!(
                map_manager, kp.id, observer_kfid); continue)

            observer_kp = get_keypoint(observer_kf, kp.id)
            observer_kp ≡ nothing && (remove_mappoint_obs!(
                map_manager, kp.id, observer_kfid); continue)

            observer_projection = project_world_to_image_distort(
                observer_kf, target_mp_position)
            avg_projection += norm(observer_kp.pixel .- observer_projection)
            n_projections += 1
        end
        avg_projection /= n_projections
        avg_projection > max_projection_distance && continue

        distance = mappoint_min_distance(target_mp, mp)
        if distance ≤ best_distance
            second_distance = best_distance
            second_id = best_id

            best_distance = distance
            best_id = kp.id
        elseif distance ≤ second_distance
            second_distance = distance
            second_id = kp.id
        end
    end

    # TODO is this necessary?
    # best_id != -1 && second_id != -1 &&
    #     0.9 * second_distance < best_distance && (best_id = -1;)

    best_id, best_distance
end

function get_new_kf!(mapper::Mapper)
    if isempty(mapper.keyframe_queue)
        mapper.new_kf_available = false
        return false, nothing
    end

    keyframe = popfirst!(mapper.keyframe_queue)
    mapper.new_kf_available = !isempty(mapper.keyframe_queue)
    true, keyframe
end

function add_new_kf!(mapper::Mapper, kf::KeyFrame)
    push!(mapper.keyframe_queue, kf)
    mapper.new_kf_available = true
end

function reset!(mapper::Mapper)
    mapper.new_kf_available = false
    mapper.exit_required = false
    mapper.keyframe_queue |> empty!
end
