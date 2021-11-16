using BSON: @save, @load
using GeometryBasics
using CairoMakie  # GLMakie
using CAB
using ProgressMeter
using FileIO

include("sfm_data.jl")


function main(n_frames)
    base_dir = "/mnt/zy_data/kitty-dataset/"
    sequence = "00"
    stereo = true

    dataset = CampusDataset(base_dir, sequence; stereo)
    println(dataset)

    save_dir = joinpath("/mnt/zy_data/kitty-dataset/projects", "kitty-$sequence")
    isdir(save_dir) || mkdir(save_dir)
    @info "Save directory: $save_dir"

    fx, fy = dataset.K[1, 1], dataset.K[2, 2]
    cx, cy = dataset.K[1:2, 3]
    height, width = 376, 1241
    # height, width = 370, 1226

    camera = CAB.Camera(fx, fy, cx, cy, 0, 0, 0, 0, height, width)
    right_camera = CAB.Camera(fx, fy, cx, cy, 0, 0, 0, 0, height, width; Ti0=dataset.Ti0)

    params = Params(;
        stereo,
        window_size=9, max_distance=35, pyramid_levels=3,
        max_nb_keypoints=1000, max_reprojection_error=3.0,
        do_local_bundle_adjustment=false, map_filtering=true)

    saver = ReplaySaver()
    visualizer = nothing
    # visualizer = Visualizer((900, 600))
    # display(visualizer)

    slam_manager = CabManager(params, camera; right_camera, visualizer=saver)
    slam_manager_thread = Threads.@spawn run!(slam_manager)

    t1 = time()
    @showprogress for i in 1:n_frames
        timestamp = dataset.timestamps[i]
        left_frame, right_frame = dataset[i]
        left_frame = Gray{Float64}.(left_frame)

        if params.stereo
            right_frame = Gray{Float64}.(right_frame)
            add_stereo_image!(slam_manager, left_frame, right_frame, timestamp)
        else
            add_image!(slam_manager, left_frame, timestamp)
        end

        if visualizer ≢ nothing
            CAB.set_image!(visualizer, rotr90(left_frame))
        end

        q_size = get_queue_size(slam_manager)
        f_size = length(slam_manager.mapper.estimator.frame_queue)
        m_size = length(slam_manager.mapper.keyframe_queue)
        while q_size > 0 || f_size > 0 || m_size > 0
            sleep(1e-2)
            q_size = get_queue_size(slam_manager)
            f_size = length(slam_manager.mapper.estimator.frame_queue)
            m_size = length(slam_manager.mapper.keyframe_queue)
        end

        sleep(1e-2)
    end

    slam_manager.exit_required = true
    wait(slam_manager_thread)

    t2 = time()
    @info "CAB took: $(t2 - t1) seconds."

    CAB.save(saver, save_dir)
    slam_manager, visualizer
end


main(4541)


