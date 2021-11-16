using StaticArrays
using LinearAlgebra
using Printf
using Images
using Manifolds
using JSON3
using PyCall

@inline function parse_matrix(line)
    m = parse.(Float64, split(line, " "))
    SMatrix{4, 4, Float64}(m..., 0, 0, 0, 1)'
end

function read_poses(poses_file)
    poses = SMatrix{4, 4, Float64}[]
    open(poses_file, "r") do reader
        while !eof(reader)
            push!(poses, parse_matrix(readline(reader)))
        end
    end
    poses
end

function read_timestamps(timestamps_file)
    timestamps = Float64[]
    open(timestamps_file, "r") do reader
        while !eof(reader)
            push!(timestamps, parse(Float64, readline(reader)))
        end
    end
    timestamps
end

# Convert XYZ to XZY.
@inbounds to_makie(positions) = [Point3f0(p[1], p[3], p[2]) for p in positions]

struct CampusDataset
    # Left camera (aka P0) intrinsic matrix. Dropped last column, which contains baselines in meters.
    Ks::Dict
    # Transformation from 0-th camera to 1-st camera.
    Ti0::SMatrix{4, 4, Float64, 16}
    # Ground truth poses. Each pose transforms from the origin.
    poses::Vector{SMatrix{4, 4, Float64, 16}}
    # Vector of timestamps for each frame.
    timestamps::Vector{Float64}
    left_frames_dir::String
    right_frames_dir::String
    stereo::Bool
end

function CampusDataset(base_dir::String, sequence::String; stereo::Bool=false)
    frames_dir = base_dir

    Ks = JSON3.read(joinpath(base_dir, "calibration_campus.json"))
    # K1 = parse_matrix(Ks[1][5:end])   # (4,4)
    K1 = Ks["0"]
    K1_inv = inv(K1)

    KT2 = parse_matrix(Ks[2][5:end])

    Ti0 = K1_inv * KT2
    Ti0 = SMatrix{4, 4, Float64}(abs(xi) < 1e-6 ? 0.0 : xi for xi in Ti0)

    timestamps = read_timestamps(joinpath(frames_dir, "times.txt"))

    left_frames_dir = joinpath(frames_dir, "image_0")
    right_frames_dir = joinpath(frames_dir, "image_1")

    poses_file = joinpath(base_dir, "poses", sequence * ".txt")
    poses = read_poses(poses_file)

    CampusDataset(K1, Ti0, poses, timestamps, left_frames_dir, right_frames_dir, stereo)
end

function get_camera_poses(dataset::CampusDataset)
    n_poses = length(dataset.poses)
    base_dir = SVector{3, Float64}(0, 0, 1)
    base_point = SVector{4, Float64}(0, 0, 0, 1)

    positions = Vector{SVector{3, Float64}}(undef, n_poses)
    directions = Vector{SVector{3, Float64}}(undef, n_poses)
    for (i, pose) in enumerate(dataset.poses)
        @inbounds positions[i] = (pose * base_point)[1:3]
        @inbounds directions[i] = normalize(pose[1:3, 1:3] * base_dir)
    end
    positions, directions
end


function collect_frames()
    while 1
        for cam_id in range(camera_num)
            image = None
            if self.output_image
                image_name = self._gen_image_name(cam_id=cam_id, this_timestamp=this_timestamp)
                image_name = os.path.join(self.root_dir, image_name)
                # image = cv2.imread(image_name)
                image = image_name  # cv2.imread(image_name)
            end
            pose_dict = dict()
            pose_dict_3d = dict()
        end

    end
end

function _gen_image_name(cam_id, timestamp)
    image_name = "Camera{0:d}/campus4-c{0:d}-{1:05d}.png".format(cam_id, timestamp)
    return image_name
end

