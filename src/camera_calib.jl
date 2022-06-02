using CameraCalibration
using CSV, DataFrames


# objp = np.zeros((w * h, 3), np.float32)
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)


function test_1()
    imgpoints = Array{Float64, 2}[]

    for i = 0:10
        data = CSV.File(string("./test/imgpoints", i, ".csv"), header = false) |> DataFrame
        push!(imgpoints, vcat(convert(Array{Float64, 2}, data)', ones(1, 35)))
    end


end


test_1()



#= 
using Plots
cord = [-ex[i][:,1:3]'*ex[i][:,4] for i in 1:11]
p = scatter([cord[i][1] for i = 1:11], [cord[i][2] for i = 1:11], [cord[i][3] for i = 1:11],marker=:circle,linewidth=0, group = 1:11)
plot(p, xlabel="X",ylabel="Y",zlabel="Z", size = (800, 800))

=#
