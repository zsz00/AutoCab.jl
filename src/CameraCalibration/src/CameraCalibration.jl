
module CameraCalibration

using Images
using ImageDraw
using Statistics, LinearAlgebra

# Includes
include("checkerboard.jl")
include("calibration.jl")

# Exports
export inner_corners, all_corners, mark_corners
export seg_boundaries_check
export check_boundaries
export process_image
export nonmaxsuppresion
export kxk_neighboardhood
export draw_dots!, draw_rect

export estimateHomography, calibrate, getHomographies, getCameraIntrinsics, getExtrinsics,
       getCameraIntrinsicsB

end  # module CameraCalib
