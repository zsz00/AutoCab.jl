
module CameraCalibration

using Images
using ImageDraw
using Statistics, LinearAlgebra

# Includes
include("checkerboard.jl")
include("calibration.jl")

# Exports
export innercorners, allcorners, markcorners
export segboundariescheck
export checkboundaries
export process_image
export nonmaxsuppresion
export kxkneighboardhood
export drawdots!, draw_rect

export estimateHomography, calibrate, getHomographies, getCameraIntrinsics, getExtrinsics,
       getCameraIntrinsicsB

end  # module CameraCalib
