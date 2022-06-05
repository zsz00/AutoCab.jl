using CameraCalibration
using CSV, DataFrames
using Images
using LazyGrids

# objp = np.zeros((w * h, 3), np.float32)
# objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imageSize, None, None)

function test_checkerboard()
    # 6*9
    dir_1 = "/home/zhangyong/codes/AutoCab.jl/src/CameraCalibration/test/capture"
    file_list = readdir(dir_1)
    println(file_list[1:end-1])
    w, h = 9,7  # 7, 5  # 9,6

    imgpoints = Array{Float64, 2}[]
    for name in file_list[1:end]
        if name == "out2"
            continue
        end
        img = Gray.(load(joinpath(dir_1, name)))
        res = process_image(img)   # n conners
        # deleteat!(res, 31);
        println(size(res), name)
        draw_rect(img, res, Gray(1));
        save(joinpath(dir_1, "out2", name), img)
        
        if size(res, 1) != 63  # 35
            continue
        end
        a = Tuple.(res)   # 
        b = vcat((hcat(i...) for i in a)...) 
        imgpoint = vcat(convert(Array{Float64, 2}, b)', ones(1, w*h))
        push!(imgpoints, imgpoint)
    end

    dh, dw = LazyGrids.ndgrid(h-1:-1:0, 0:w-1)
    obj_points = vcat(reshape(dw',1,:), reshape(dh',1,:), ones(1, w*h))
    
    aInit, wInit = calibrate(obj_points, imgpoints) # 不通
end


test_checkerboard()



#= 

"""
tests
asample = [182 535 171 537; 350 358 553 563; 1 1 1 1]
bsample = [0 888 0 888; 0 0 500 500; 1 1 1 1]
7*5
testobj = [0. 1. 2. 3. 4. 5. 6. 0. 1. 2. 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6 0 1 2 3 4 5 6;
           4 4 4 4 4 4 4 3 3 3 3 3 3 3 2 2 2 2 2 2 2 1 1 1 1 1 1 1 0 0 0 0 0 0 0;
           1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

=#
