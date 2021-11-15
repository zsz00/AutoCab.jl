"""
Map Manager is responsible for managing Keyframes in the map
as well as Mappoints.
地图管理器负责管理地图中的关键帧以及地图点.

# Arguments:

- `current_frame::Frame`: Current frame that is shared throughout
    all the components in the system.
- `frames_dict::Dict{Int64, Frame}`: Dict of the Keyframes (its id → Keyframe).
- `params::Params`: Parameters of the system.
- `map_points_dict::Dict{Int64, MapPoint}`: Map of all the map_points
    (its id → MapPoints).
- `current_mappoint_id::Int64`: Id of the current Mappoint to be created.
    It is incremented each time a new Mappoint is added to `map_points`.
- `current_keyframe_id::Int64`: Id of the current Keyframe to be created.
    It is incremented each time a new Keyframe is added to `frames_map`.
- `nb_keyframes::Int64`: Total number of keyframes.
- `nb_mappoints::Int64`: Total number of mappoints.
"""
mutable struct MapManager
    params::Params
    current_frame::Frame
    frames_dict::Dict{Int64, Frame}   # 所有的frames
    map_points_dict::Dict{Int64, MapPoint}   # 所有的3d points 
    
    current_mappoint_id::Int64
    current_keyframe_id::Int64
    nb_keyframes::Int64
    nb_mappoints::Int64
end

function MapManager(params::Params, frame::Frame)
    MapManager(
        params, frame, Dict{Int64, Frame}(), 
        Dict{Int64, MapPoint}(), 0, 0, 0, 0)
end


function get_keyframe(m::MapManager, kfid)
    get(m.frames_dict, kfid, nothing)
end

function has_keyframe(m::MapManager, kfid)
    kfid in keys(m.frames_dict)
end

function get_mappoint(m::MapManager, mpid)
    get(m.map_points_dict, mpid, nothing)
end

function create_keyframe!(m::MapManager, image)
    @debug "[MM] Creating new keyframe $(m.current_keyframe_id)."
    prepare_frame!(m)
    extract_keypoints!(m, image)
    add_keyframe!(m)
end

function prepare_frame!(m::MapManager)
    m.current_frame.kfid = m.current_keyframe_id
    @debug "[MM] Adding KF $(m.current_frame.kfid) to Map."

    # Filter if there are too many keypoints.
    # if m.current_frame.nb_keypoints > m.params.max_nb_keypoints
    #     # TODO
    # end

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

        prev_observers = get_observers(prev_mp)
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
        display(stacktrace(catch_backtrace())); println()
    end
end

# 光流匹配
function optical_flow_matching!(
    map_manager::MapManager, frame::Frame,
    from_pyramid::LKPyramid, to_pyramid::LKPyramid, stereo,
)
    window_size = map_manager.params.window_size
    max_distance = map_manager.params.max_ktl_distance
    pyramid_levels = map_manager.params.pyramid_levels

    pyramid_levels_3d = 1
    ids = Vector{Int64}(undef, frame.nb_keypoints)
    pixels = Vector{Point2f}(undef, frame.nb_keypoints)

    ids3d = Vector{Int64}(undef, frame.nb_3d_kpts)
    pixels3d = Vector{Point2f}(undef, frame.nb_3d_kpts)
    displacements3d = Vector{Point2f}(undef, frame.nb_3d_kpts)

    i, i3d = 1, 1
    scale = 1.0 / 2.0^pyramid_levels_3d
    n_good = 0

    keypoints = stereo ? get_keypoints(frame) : values(frame.keypoints)
    for kp in keypoints
        if !kp.is_3d
            pixels[i] = kp.pixel
            ids[i] = kp.id
            i += 1
            continue
        end

        mp = stereo ?
            get_mappoint(map_manager, kp.id) :
            map_manager.map_points_dict[kp.id]
        if mp ≡ nothing
            remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
            continue
        end

        position = get_position(mp)
        projection = stereo ?
            project_world_to_right_image_distort(frame, position) :
            project_world_to_image_distort(frame, position)

        if stereo
            if in_right_image(frame, projection)
                ids3d[i3d] = kp.id
                pixels3d[i3d] = kp.pixel
                displacements3d[i3d] = scale .* (projection .- kp.pixel)
                i3d += 1
            else
                remove_mappoint_obs!(map_manager, kp.id, frame.kfid)
                continue
            end
        else
            if in_image(frame, projection)
                ids3d[i3d] = kp.id
                pixels3d[i3d] = kp.pixel
                displacements3d[i3d] = scale .* (projection .- kp.pixel)
                i3d += 1
            end
        end
    end

    i3d -= 1
    ids3d = @view(ids3d[1:i3d])
    pixels3d = @view(pixels3d[1:i3d])
    displacements3d = @view(displacements3d[1:i3d])

    failed_3d = true
    if !isempty(ids3d)
        new_keypoints, status = fb_tracking!(
            from_pyramid, to_pyramid, pixels3d;
            displacement=displacements3d,
            pyramid_levels=pyramid_levels_3d,
            window_size, max_distance)

        nb_good = 0
        for j in 1:length(status)
            if status[j]
                update_keypoint!(frame, ids3d[j], new_keypoints[j])
                nb_good += 1
            else
                # If failed → add to track with 2d keypoints w/o prior.
                pixels[i] = pixels3d[j]
                ids[i] = ids3d[j]
                i += 1
            end
        end
        @debug "[MM] 3D Points tracked $nb_good. Stereo $stereo."
        failed_3d = nb_good < 0.33 * length(ids3d)
    end

    i -= 1
    pixels = @view(pixels[1:i])
    ids = @view(ids[1:i])

    isempty(pixels) && return nothing
    new_keypoints, status = fb_tracking!(
        from_pyramid, to_pyramid, pixels;
        pyramid_levels, window_size, max_distance)

    for j in 1:length(new_keypoints)
        status[j] ? update_keypoint!(frame, ids[j], new_keypoints[j]) :
                remove_obs_from_current_frame!(map_manager, ids[j])

    end
    nothing
end
