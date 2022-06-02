using CameraCalibration
using CSV, DataFrames
using Images

# objp = np.zeros((w * h, 3), np.float32)
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

function test_checkerboard()
    img = Gray.(load("/home/zhangyong/codes/AutoCab.jl/src/CameraCalibration/test/test.jpg"))
    res = process_image(img)
    draw_rect(img, res,  Gray(1))
    save("results.png", img)
end


function test_calibration()
    imgpoints = Array{Float64, 2}[]

    for i = 0:10
        data = CSV.File(string("./test/imgpoints", i, ".csv"), header = false) |> DataFrame
        push!(imgpoints, vcat(convert(Array{Float64, 2}, data)', ones(1, 35)))
    end

end


test_calibration()



#= 

"""
tests
asample = [182 535 171 537; 350 358 553 563; 1 1 1 1]
bsample = [0 888 0 888; 0 0 500 500; 1 1 1 1]

testobj = [0. 1. 2. 3. 4. 5. 6. 0. 1. 2. 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6;
           4 4 4 4 4 4 4 3 3 3 3 3 3 3 2 2 2 2 2 2 2 1 1 1 1 1 1 1 0 0 0 0 0 0 0;
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
~
using CSV, DataFrames
imgpoints = Array{Float64, 2}[]
    for i = 0:10
        push!(imgpoints, vcat(convert(Array{Float64, 2}, CSV.File(string("./test/imgpoints", i, ".csv"), header = false) |> DataFrame)', ones(1, 35)))
    end

using Plots
cord = [-ex[i][:,1:3]'*ex[i][:,4] for i in 1:11]
p = scatter([cord[i][1] for i = 1:11], [cord[i][2] for i = 1:11], [cord[i][3] for i = 1:11],marker=:circle,linewidth=0, group = 1:11)
plot(p, xlabel="X",ylabel="Y",zlabel="Z", size = (800, 800))
"""
=#
