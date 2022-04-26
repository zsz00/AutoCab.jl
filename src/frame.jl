"""
Keypoint is a feature in the image used for tracking.
关键点是图像中用于跟踪的一个特征. 

# Parameters:

- `id::Int64`: Id of the Keypoint.
- `pixel::Point2f`: Pixel coordinate in the image plane in `(y, x)` format.
- `undistorted_pixel::Point2f`: In presence of distortion in camera,
    this is the undistorted `pixel` coordinates.
- `position::Point3f`: Pre-divided (backprojected) keypoint in camera space.
    This is used in algorithms like 5Pt for Essential matrix calculation.
- `descriptor::BitVector`: Descriptor of a keypoint.
- `is_3d::Bool`: If `true`, then this keypoint was triangulated.
- `is_retracked::Bool`: If `true`, then this keypoint was lost
    and re-tracked back by `match_local_map!` method.
"""
mutable struct Keypoint
    id::Int64
    pixel::Point2f
    undistorted_pixel::Point2f   # 去畸变的 像素坐标点
    position::Point3f   # 相机坐标的3d点位置
    point_type::Int64  # person landmark, 物体sift
    descriptor::BitVector   # 点的描述子
    person_id::Int64    
    lmkpoint_id::Int8   # landmark point id
    is_3d::Bool   
    is_retracked::Bool  # 再跟踪
end

function Keypoint(id, pixel, undistorted_pixel, position, descriptor, person_id, lmkpoint_id, is_3d)
    Keypoint(id, pixel, undistorted_pixel, position, descriptor, person_id, lmkpoint_id, is_3d, false)
end

"""
Frame that encapsulates information of a camera in space at a certain moment.

# Parameters:

- `id::Int64`: Id of the Frame.
- `kfid::Int64`: Id of the Frame in the global map, contained in MapManager.
    Frames that are in this map are called "Key-Frames".
- `time::Float64`: Time of the frame at which it was taken on the camera.
- `camera_id::Int64`: Camera id associated with this frame.
- `keypoints::Dict{Int64, Keypoint}`: Map of that this frames observes.
- `ketpoints_grid::Matrix{Set{Int64}}`: Grid, where each cell contains
    several keypoints. This is useful when want to retrieve neighbours
    for a certain Keypoint.
- `nb_occupied_cells::Int64`: Number of cells in `keypoints_grid` that have
    at least one Keypoint.
- `cell_size::Int64`: Cell size in pixels.
- `nb_keypoints::Int64`: Total number of keypoints in the Frame.
- `nb_2d_keypoints::Int64`: Total number of 2D keypoints in the Frame.
- `nb_3d_keypoints::Int64`: Total number of 3D keypoints in the Frame.
- `covisible_kf::OrderedDict{Int64, Int64}`: Dictionary with `kfid` => `score`
    of ids of Frames that observe the sub-set of size `score` of keypoints in Frame.
- `local_map_ids::Set{Int64}`: Set of ids of MapPoints that are not visible
    in this Frame, but are a potential candidates for remapping
    back into this Frame.
"""
mutable struct Frame
    id::Int64   # frame id
    kfid::Int64  # Key-Frame id

    time::Float64
    camera_id::Int64
    keypoints::Dict{Int64, Keypoint}
    keypoints_grid::Matrix{Set{Int64}}

    nb_occupied_cells::Int64
    cell_size::Int64

    nb_keypoints::Int64 # Total number of keypoints in the Frame.
    nb_2d_kpts::Int64  # Total number of 2D keypoints in the Frame.
    nb_3d_kpts::Int64  # Total number of 3D keypoints in the Frame.

    covisible_kf::OrderedDict{Int64, Int64}   # 可变的kframe
    local_map_ids::Set{Int64}
end

function Frame(; camera_id, cell_size, id = 0, kfid = 0, time = 0.0)
    nb_keypoints = 0
    nb_occupied_cells = 0
    keypoints = Dict{Int64, Keypoint}()

    image_resolution = (camera.height, camera.width)
    cells = ceil.(Int64, image_resolution ./ cell_size)
    grid = [Set{Int64}() for _=1:cells[1], _=1:cells[2]]

    Frame(
        id, kfid, time, 
        camera_id, keypoints, grid,
        nb_occupied_cells, cell_size,
        nb_keypoints, 0, 0, 0,
        OrderedDict{Int64, Int64}(), Set{Int64}()
    )
end


function get_keypoints(f::Frame)
    [deepcopy(kp) for kp in values(f.keypoints)]
end

function get_2d_keypoints(f::Frame)
    kps = Vector{Keypoint}(undef, f.nb_2d_kpts)
    i = 1
    for k in values(f.keypoints)
        k.is_3d || (kps[i] = deepcopy(k); i += 1;)
    end
    @assert (i - 1) == f.nb_2d_kpts
    kps
end

function get_3d_keypoints(f::Frame)
    kps = Vector{Keypoint}(undef, f.nb_3d_kpts)
    i = 1
    for k in values(f.keypoints)
        k.is_3d && (kps[i] = deepcopy(k); i += 1;)
    end
    @assert (i - 1) == f.nb_3d_kpts
    kps
end

function get_3d_keypoints_nb(f::Frame)
    return f.nb_3d_kpts
end

function get_3d_keypoints_ids(f::Frame)
    ids = Vector{Int64}(undef, f.nb_3d_kpts)
    i = 1
    for k in values(f.keypoints)
        k.is_3d && (ids[i] = k.id; i += 1;)
    end
    @assert (i - 1) == f.nb_3d_kpts
    ids
end

function get_keypoint(f::Frame, kpid)
    deepcopy(get(f.keypoints, kpid, nothing))
end

function get_keypoint_unpx(f::Frame, kpid)
    kpid in keys(f.keypoints) ? f.keypoints[kpid].undistorted_pixel : nothing

end

function add_keypoint!(f::Frame, point, id; descriptor::BitVector = BitVector(), is_3d::Bool = false,)
    undistorted_point = undistort_point(f.camera, point)  # 2d point to undistort_point
    position = backproject(f.camera, undistorted_point)  # 2d undistort_point to 3d point (x,y,z=1).
    add_keypoint!(f, Keypoint(id, point, undistorted_point, position, descriptor, is_3d))
end

function add_keypoint!(f::Frame, keypoint::Keypoint)
    if keypoint.id in keys(f.keypoints)
        @warn "[Frame] $(f.id) already has keypoint $(keypoint.id)."
        return
    end

    f.keypoints[keypoint.id] = keypoint
    add_keypoint_to_grid!(f, keypoint)

    f.nb_keypoints += 1
    if keypoint.is_3d
        f.nb_3d_kpts += 1
    else
        f.nb_2d_kpts += 1
    end
end

function update_keypoint!(f::Frame, kpid, point)
    ckp = get(f.keypoints, kpid, nothing)
    ckp ≡ nothing && return

    kp = ckp |> deepcopy
    kp.pixel = point
    kp.undistorted_pixel = undistort_point(f.camera, kp.pixel)
    kp.position = backproject(f.camera, kp.undistorted_pixel)

    update_keypoint_in_grid!(f, ckp, kp)
    f.keypoints[kpid] = kp
end

function update_keypoint!(f::Frame, prev_id, new_id, is_3d)
    has_new = false
    has_new = new_id in keys(f.keypoints)
    has_new && return false

    prev_kp = get_keypoint(f, prev_id)
    prev_kp ≡ nothing && return false

    prev_kp.id = new_id
    prev_kp.is_retracked = true
    prev_kp.is_3d = is_3d

    remove_keypoint!(f, prev_id)
    add_keypoint!(f, prev_kp)
    true
end

function update_keypoint_in_grid!(
    f::Frame, previous_keypoint::Keypoint, new_keypoint::Keypoint,
)
    prev_kpi = to_cartesian(previous_keypoint.pixel, f.cell_size)
    new_kpi = to_cartesian(new_keypoint.pixel, f.cell_size)
    prev_kpi == new_kpi && return
    # Update grid, when new keypoint changes its position
    # so much as to move to the other grid cell.
    remove_keypoint_from_grid!(f, previous_keypoint)
    add_keypoint_to_grid!(f, new_keypoint)
end

function add_keypoint_to_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells += 1;)
    push!(f.keypoints_grid[kpi], keypoint.id)
end

function remove_keypoint_from_grid!(f::Frame, keypoint::Keypoint)
    kpi = to_cartesian(keypoint.pixel, f.cell_size)
    if keypoint.id in f.keypoints_grid[kpi]
        pop!(f.keypoints_grid[kpi], keypoint.id)
        isempty(f.keypoints_grid[kpi]) && (f.nb_occupied_cells -= 1)
    end
end

function remove_keypoint!(f::Frame, kpid)
    kp = get(f.keypoints, kpid, nothing)
    kp ≡ nothing && return

    pop!(f.keypoints, kpid)
    remove_keypoint_from_grid!(f, kp)

    f.nb_keypoints -= 1
    if kp.is_3d
        f.nb_3d_kpts -= 1
    else
        f.nb_2d_kpts -= 1
    end
end

function turn_keypoint_3d!(f::Frame, id)
    kp = get(f.keypoints, id, nothing)
    kp ≡ nothing && return
    kp.is_3d && return

    kp.is_3d = true
    f.nb_2d_kpts -= 1
    f.nb_3d_kpts += 1
end

function get_covisible_map(f::Frame)
    deepcopy(f.covisible_kf)
end

function set_covisible_map!(f::Frame, covisible_map)
    f.covisible_kf = covisible_map
end

function add_covisibility!(f::Frame, kfid, cov_score)
    kfid == f.kfid && return
    f.covisible_kf[kfid] = cov_score
end

function add_covisibility!(f::Frame, kfid)
    kfid == f.kfid && return
    score = get(f.covisible_kf, kfid, 0)
    f.covisible_kf[kfid] = score + 1
end

function decrease_covisible_kf!(f::Frame, kfid)
    kfid == f.kfid && return
    cov_score = get(f.covisible_kf, kfid, nothing)
    (cov_score ≡ nothing || cov_score == 0) && return

    cov_score -= 1
    f.covisible_kf[kfid] = cov_score
    cov_score == 0 && pop!(f.covisible_kf, kfid)
end

function remove_covisible_kf!(f::Frame, kfid)
    kfid == f.kfid && return
    kfid in keys(f.covisible_kf) && pop!(f.covisible_kf, kfid)
end

function is_observing_kp(f::Frame, kpid)
    kpid in keys(f.keypoints)
end

function get_surrounding_keypoints(f::Frame, kp::Keypoint)
    keypoints = Vector{Keypoint}(undef, 0)
    sizehint!(keypoints, 20)
    kpi = to_cartesian(kp.pixel, f.cell_size)

    for r in (kpi[1] - 1):(kpi[1] + 1), c in (kpi[2] - 1):(kpi[2] + 1)
        (r < 1 || c < 1 || r > size(f.keypoints_grid, 1)
            || c > size(f.keypoints_grid, 2)) && continue

        for cell_kpid in f.keypoints_grid[r, c]
            cell_kpid == kp.id && continue
            cell_kp = get(f, cell_kpid, nothing)
            cell_kp ≡ nothing || push!(keypoints, cell_kp)
        end
    end

    keypoints
end

function get_surrounding_keypoints(f::Frame, pixel)
    keypoints = Vector{Keypoint}(undef, 0)
    sizehint!(keypoints, 20)
    kpi = to_cartesian(pixel, f.cell_size)

    for r in (kpi[1] - 1):(kpi[1] + 1), c in (kpi[2] - 1):(kpi[2] + 1)
        (r < 1 || c < 1 || r > size(f.keypoints_grid, 1)
            || c > size(f.keypoints_grid, 2)) && continue

        for cell_kpid in f.keypoints_grid[r, c]
            cell_kp = get(f.keypoints, cell_kpid, nothing)
            cell_kp ≡ nothing || push!(keypoints, cell_kp)
        end
    end

    keypoints
end

in_image(f::Frame, point) = in_image(f.camera, point)