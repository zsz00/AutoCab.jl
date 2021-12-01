"""
Camera

# Parameters:
- `id::Int64`: Id of the Camera.
- `cw::SMatrix{4, 4, Float64, 16}`: Transformation matrix `[R|t]` that transforms from world to camera space.
- `wc::SMatrix{4, 4, Float64, 16}`: Transformation matrix `[R|t]` that transforms from camera to world space.
"""
mutable struct Camera
    id::Int64
    # Focal length.
    fx::Float64
    fy::Float64
    # Principal point.
    cx::Float64
    cy::Float64
    # Radial lens distortion.
    k1::Float64
    k2::Float64
    # Tangential lens distortion.
    p1::Float64
    p2::Float64
    # Intrinsic matrix.
    K::SMatrix{3,3,Float64,9}
    iK::SMatrix{3,3,Float64,9}
    # Image resolution.
    height::Int64
    width::Int64

    # Transformation from 0-th camera to i-th (this).  w to c
    Ti0::SMatrix{4,4,Float64,16}
    # Transformation from i-th (this) camera to 0-th.
    T0i::SMatrix{4,4,Float64,16}

    cw::SMatrix{4, 4, Float64, 16}   # pose, w to c
    wc::SMatrix{4, 4, Float64, 16}   # pose, c to w
end

function Camera(
    id, fx, fy, cx, cy, # Intrinsics.
    k1, k2, p1, p2, # Distortion coefficients.
    height, width; Ti0=SMatrix{4,4,Float64}(I),
)
    K = SMatrix{3,3,Float64,9}(
        fx, 0.0, 0.0,
        0.0, fy, 0.0,
        cx, cy, 1.0)
    iK = K |> inv

    Camera(id, fx, fy, cx, cy, k1, k2, p1, p2,
           K, iK, height, width, Ti0, inv(SE3, Ti0), Ti0, Ti0)
end

"""
Project point from 3D space onto the image plane.
投影变换
# Arguments:
- `point`: Point in 3D space in `(x, y, z)` format. 相机坐标系的

# Returns:
Projected point in `(y, x)` format.
"""
function project(c::Camera, point)
    inv_z = 1.0 / point[3]
    Point2f(c.fy * point[2] * inv_z + c.cy, c.fx * point[1] * inv_z + c.cx)
end

"""
Project `point` onto image plane of the `Camera`,
accounting for the distortion parameters of the camera.

# Arguments:
- `point`: 3D point to project in `(x, y, z)` format.

# Returns:
2D floating point coordinates in `(y, x)` format.
"""
function project_undistort(c::Camera, point)
    normalized = point[2:-1:1] ./ point[3]
    undistort_pdn_point(c, normalized)
end

"""
Check if `point` is in the image bounds of the `Camera`.

# Arguments:
- `point`: Point to check. In `(y, x)` format.
"""
function in_image(c::Camera, point)
    1 ≤ point[1] ≤ c.height && 1 ≤ point[2] ≤ c.width
end

"""
# Arguments:
- `point::SVector{2}`: Point to undistort. In `(y, x)` format.
"""
function undistort_point(c::Camera, point::Point2)
    normalized = Point2f(
        (point[1] - c.cy) / c.fy,
        (point[2] - c.cx) / c.fx)
    undistort_pdn_point(c, normalized)
end

"""
Undistort point.

# Arguments:
- `point::SVector{2}`: Predivided by `K` & normalized point in `(y, x)` format.
"""
function undistort_pdn_point(c::Camera, point)
    sqrd_normalized = point.^2
    # Square radius from center.
    sqrd_radius = sqrd_normalized |> sum
    # Radial distortion factor.
    rd = 1.0 + c.k1 * sqrd_radius + c.k2 * sqrd_radius^2
    # Tangential distortion component.
    p = point |> prod
    dtx = 2 * c.p1 * p + c.p2 * (sqrd_radius + 2 * sqrd_normalized[1])
    dty = c.p1 * (sqrd_radius + 2 * sqrd_normalized[2]) + 2 * c.p2 * p
    # Lens distortion coordinates.
    distorted = rd .* point .+ (dty, dtx)
    # Final projection (assume skew is always `0`).
    Point2f(distorted .* (c.fy, c.fx) .+ (c.cy, c.cx))
end

"""
Transform point from 2D to 3D by dividing by `K`.

# Arguments:

- `point::Point2`: Point to backproject in `(y, x)` format.

# Returns:

Backprojected `Point3f` in `(x, y, z = 1.0)` format.
"""
function backproject(c::Camera, point::Point2)
    Point3f((point[2] - c.cx) / c.fx, (point[1] - c.cy) / c.fy, 1.0)
end


function set_wc!(c::Camera, wc, visualizer = nothing)
    c.wc = wc
    c.cw = inv(SE3, wc)
    visualizer ≢ nothing
end

function set_cw!(c::Camera, cw, visualizer = nothing)
    c.cw = cw
    c.wc = inv(SE3, cw)   # 用李群李代数 来做这个变换. 
    visualizer ≢ nothing
end

function get_cw(c::Camera)
    c.cw
end

function get_wc(c::Camera)
    c.wc
end

function get_Rwc(c::Camera)
    c.wc[1:3, 1:3]
end

function get_Rcw(c::Camera)
    c.cw[1:3, 1:3]
end

function get_twc(c::Camera)
    c.wc[1:3, 4]
end

function get_tcw(c::Camera)
    c.cw[1:3, 4]
end

function get_cw_ba(c::Camera)::NTuple{6, Float64}
    r = RotZYX(c.cw[1:3, 1:3])   # r_vec
    (r.theta1, r.theta2, r.theta3, c.cw[1:3, 4]...)   # r_vec,t
end

function get_wc_ba(c::Camera)::NTuple{6, Float64}
    r = RotXYZ(c.wc[1:3, 1:3])
    (r.theta1, r.theta2, r.theta3, c.wc[1:3, 4]...)
end

function set_cw_ba!(c::Camera, θ)
    set_cw!(c, to_4x4(RotZYX(θ[1:3]...), θ[4:6]))
end

function project_camera_to_world(c::Camera, point)
    c.wc * to_homogeneous(point)
end

function project_world_to_camera(c::Camera, point)
    # 世界坐标系到相机坐标系. point是(x,y,z)
    c.cw * to_homogeneous(point)
end

function project_world_to_image(c::Camera, point)
    # 世界坐标系到图片坐标系
    project(c, project_world_to_camera(c, point))
end

function project_world_to_image_distort(c::Camera, point)
    project_undistort(c, project_world_to_camera(c, point))
end
