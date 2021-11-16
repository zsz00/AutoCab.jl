"""
Map Manager is responsible for managing Keyframes in the map
as well as Mappoints.
地图管理器负责管理地图中的关键帧以及地图点.
"""
mutable struct MapManager
    current_frame::Frame
    frames_dict::Dict{Int64, Frame}   # 所有的frames
    map_points_dict::Dict{Int64, MapPoint}   # 所有的3d points 
    
    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_keyframes::Int64
    nb_mappoints::Int64
end

function MapManager(frame::Frame)
    MapManager(frame, Dict{Int64, Frame}(), Dict{Int64, MapPoint}(), 0, 0, 0, 0)
end

function create_keyframe!(m::MapManager, image)
    @debug "[MM] Creating new keyframe $(m.current_keyframe_id)."
    prepare_frame!(m)   # 预处理frame
    extract_keypoints!(m, image)   # 提取特征点, add_keypoints_to_frame. 
    add_keyframe!(m)   # 把frame加入到MapManager
end

function prepare_frame!(m::MapManager)
    m.current_frame.kfid = m.current_keyframe_id
    @debug "[MM] Adding KF $(m.current_frame.kfid) to Map."

    for kp in values(m.current_frame.keypoints)
        mp = get(m.map_points_dict, kp.id, nothing)
        if mp ≡ nothing
            remove_obs_from_current_frame!(m, kp.id)
        else
            add_keyframe_observation!(mp, m.current_keyframe_id)
        end
    end
end

function add_keypoints_to_frame!(m::MapManager, frame, keypoints, descriptors)
    for (kp, dp) in zip(keypoints, descriptors)
        # m.current_mappoint_id is incremented in `add_mappoint!`.
        add_keypoint!(frame, Point2f(kp[1], kp[2]), m.current_mappoint_id)
        add_mappoint!(m, dp)
    end
end

@inline function add_mappoint!(m::MapManager, descriptor)
    mp = MapPoint(m.current_mappoint_id, m.current_keyframe_id, descriptor)
    m.map_points_dict[m.current_mappoint_id] = mp
    m.current_mappoint_id += 1
    m.nb_mappoints += 1
end

function add_keyframe!(m::MapManager)
    new_keyframe = deepcopy(m.current_frame)
    m.frames_dict[m.current_keyframe_id] = new_keyframe
    m.current_keyframe_id += 1
    m.nb_keyframes += 1
end

function update_mappoint!(m::MapManager, mpid, new_position)
    mp = m.map_points_dict[mpid]
    # If MapPoint is 2D, turn it to 3D and update its observing KeyFrames.
    if !mp.is_3d
        for observer_id in mp.observer_keyframes_ids
            if observer_id in keys(m.frames_dict)
                turn_keypoint_3d!(m.frames_dict[observer_id], mpid)
            else
                remove_kf_observation!(mp, observer_id)
            end
        end
        if mp.is_observed
            # Because we deepcopy frame before putting it to the frames_map,
            # we need to update current frame as well.
            # Which should also update current frame in the FrontEnd.
            turn_keypoint_3d!(m.current_frame, mpid)
        end
    end
    set_position!(mp, new_position)

end

function update_frame_covisibility!(map_manager::MapManager, frame::Frame)
    covisible_keyframes = Dict{Int64, Int64}()
    local_map_ids = Set{Int64}()
    # For each Keypoint in the `frame`, get its corresponding MapPoint.
    # Get the set of KeyFrames that observe this MapPoint.
    # Add them to the covisible map, which contains all KeyFrames
    # that share visibility with the `frame`.
    for kp in get_keypoints(frame)
        mp = get_mappoint(map_manager, kp.id)
        mp_observers = get_observers(mp)
        # Get the set of KeyFrames observing this KeyFrame to update covisibility.
        for kfid in mp_observers
            kfid == frame.kfid && continue
            if kfid in keys(covisible_keyframes)
                covisible_keyframes[kfid] += 1
            else
                covisible_keyframes[kfid] = 1
            end
        end
    end
    # Update covisibility for covisible KeyFrames.
    # For each entry in the covisible map, get its corresponding KeyFrame.
    # Update the covisible score for the `frame` in it.
    # Add all 3D Keypoints that are not in the `frame`
    # to the local map for future tracking.
    for (kfid, cov_score) in covisible_keyframes
        cov_frame = get_keyframe(map_manager, kfid)
        add_covisibility!(cov_frame, frame.kfid, cov_score)
        for kp in get_3d_keypoints(cov_frame)
            kp.id in keys(frame.keypoints) || push!(local_map_ids, kp.id)
        end
    end

    # Update the set of covisible KeyFrames.
    set_covisible_map!(frame, covisible_keyframes)
    # Update local map of unobserved MapPoints.
    if length(local_map_ids) > 0.5 * length(frame.local_map_ids)
        frame.local_map_ids = local_map_ids
    else
        union!(frame.local_map_ids, local_map_ids)
    end
end

function merge_mappoints(m::MapManager, prev_id, new_id)
    try
        prev_mp = get(m.map_points_dict, prev_id, nothing)
        prev_mp ≡ nothing && return
        new_mp = get(m.map_points_dict, new_id, nothing)
        new_mp ≡ nothing && return
        new_mp.is_3d || return

        prev_observers = get_observers(prev_mp)  # 取出此3d点的所有 观测frames id
        new_observers = get_observers(new_mp)

        # For previous mappoint observers, update keypoint for them.
        # If successfull, then add covisibility link between old and new observer keyframes.
        for prev_observer_id in prev_observers
            prev_observer_kf = get(m.frames_dict, prev_observer_id, nothing)
            prev_observer_kf ≡ nothing && continue
            update_keypoint!(prev_observer_kf, prev_id, new_id, new_mp.is_3d) || continue

            add_keyframe_observation!(new_mp, prev_observer_id)
            for new_observer_id in new_observers
                new_observer_kf = get(m.frames_dict, new_observer_id, nothing)
                new_observer_kf ≡ nothing && continue

                add_covisibility!(new_observer_kf, prev_observer_id)
                add_covisibility!(prev_observer_kf, new_observer_id)
            end
        end

        for (kfid, descriptor) in prev_mp.keyframes_descriptors
            add_descriptor!(new_mp, kfid, descriptor)
        end
        if is_observing_kp(m.current_frame, prev_id)
            update_keypoint!(m.current_frame, prev_id, new_id, new_mp.is_3d)
        end

        # Update nb mappoints and erase old mappoint.
        prev_mp.is_3d && (m.nb_mappoints -= 1;)
        pop!(m.map_points_dict, prev_id)
    catch e
        showerror(stdout, e); println()
    end
end


