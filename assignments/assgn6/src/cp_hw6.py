import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#images: list of filenames for checkboard calibration files
#checkerboard: dimension of the inner corners
#dW: corner refinement window size. should be smaller for lower resolution images
def computeIntrinsic(images, checkerboard, dW):
    # Defining the dimensions of checkerboard
    # Number of inner corners, hard coded to class example
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    img_shape = None

    # Extracting path of individual image stored in a given directory
    print('Displaying chessboard corners. Press any button to continue to next example')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        # TODO: touchup
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            #print(corners)
            corners2 = cv2.cornerSubPix(gray, corners, dW, (-1,-1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        else:
            print("error: checkerboard not found")

        cv2.imshow('img',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
        ret:
        mtx: camera matrix
        dist: distortion coefficients

    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("Camera matrix: \n")
    print(mtx)
    print("Distortion: \n")
    print(dist)

    return mtx, dist


#HACK global variables for callbacks… fix this someday…
X_CAPT = np.float32([])
Y_CAPT = np.float32([])
def computeExtrinsic(img_path, mtx, dist, dX, dY):
    global X_CAPT, Y_CAPT
    X_CAPT = np.float32([])
    Y_CAPT = np.float32([])
    color_img = cv2.imread(img_path)
    I = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.001)

    def capture_click(event, x_click, y_click, flags, params):
        global X_CAPT, Y_CAPT
        if event == cv2.EVENT_LBUTTONDOWN:
            xy_click = np.float32([x_click, y_click])
            xy_click = xy_click.reshape(-1, 1, 2)
            #print(xy_click)
            refined_xy = cv2.cornerSubPix(I, xy_click, (11, 11), (-1, -1), criteria)
            #print(refined_xy)
            X_CAPT = np.append(X_CAPT, refined_xy[0, 0, 0])
            Y_CAPT = np.append(Y_CAPT, refined_xy[0, 0, 1])
            cv2.drawMarker(color_img, (int(X_CAPT[-1]), int(Y_CAPT[-1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30)

    #plt.imshow(color_img[:,:,::-1])
    compute_name = 'Define Extrinsic'
    cv2.namedWindow(compute_name)
    cv2.setMouseCallback(compute_name, capture_click)

    print("Click on the four corners of the rectangular pattern, starting from the bottom-left and proceeding counter-clockwise.")
    #draw user selected features
    while True:
        cv2.imshow(compute_name, color_img)
        key = cv2.waitKey(1)

        if key == ord("q") or len(X_CAPT) == 4:
            break
    x = X_CAPT
    y = Y_CAPT
    #print("click input: ", x, y)

    #sort corners
    #x_v = x - x.mean()
    #y_v = y - y.mean()
    #theta = np.arctan2(-y_v, x_v)
    #ind = np.argsort(np.mod(theta-theta[0], 2*np.pi))
    #ind = ind[::-1]
    #x = x[ind]
    #y = y[ind]
    #print ("sorted corners: ", x, y)

    #xy_corners_undist = cv2.undistortPoints(, mtx, dist )

    #project grid points
    #M = cv2.findhomography()
    #distort projected points

    img_points = np.vstack((x, y)).T.reshape(-1, 1, 2)
    obj_points = np.array([[0, dX, dX, 0], [0, 0, dY, dY], [0, 0, 0, 0]]).T.reshape(-1, 1, 3)
    #print("image points: ", img_points)
    #print("object points: ", obj_points)
    result = cv2.solvePnPRansac(obj_points, img_points, mtx, dist)
    #print(result)
    rvec = result[1]
    tvec = result[2]
    rmat = cv2.Rodrigues(rvec)[0]


    #display axis
    #origin, x (red), y (green), z (blue)
    axis = np.float32([[0,0,0], [dX,0,0], [0,dY,0], [0,0,min(dX, dY)]]).reshape(-1,3)
    axis_img = cv2.projectPoints(axis, rvec, tvec, mtx, dist)[0]
    #print("projected points: ", axis_img)

    axis_img = axis_img.astype(int)

    #cv2.namedWindow('Result')
    cv2.putText(color_img, 'X', (axis_img[1,0,0], axis_img[1,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[1,0,0], axis_img[1,0,1]), (0,0,255), 2) #x
    cv2.putText(color_img, 'Y', (axis_img[2,0,0], axis_img[2,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[2,0,0], axis_img[2,0,1]), (0,255,0), 2) #y
    cv2.putText(color_img, 'Z', (axis_img[3,0,0], axis_img[3,0,1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0))
    cv2.line(color_img, (axis_img[0,0,0], axis_img[0,0,1]), (axis_img[3,0,0], axis_img[3,0,1]), (255,0,0), 2) #z
    cv2.imshow(compute_name, color_img)
    print("Done! Press any key to exit")

    cv2.waitKey(0)

    #visualize camera relative to calibration plane
    image_corners = np.float32([[0,0], [I.shape[1], 0], [0, I.shape[0]], [I.shape[1], I.shape[0]]])

    corner_colors = [(0,0,0), (1,0,1), (0,0,1), (0,0,1)]
    print("image corners: ", image_corners, image_corners.shape)
    corner_rays = np.matmul(rmat.T, np.squeeze(200*pixel2ray(image_corners, mtx, dist)).T)
    print(corner_rays)
    fig = plt.figure("Projected camera view")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot([0, dX, dX, 0, 0], [0, 0, dY, dY, 0])
    C = np.matmul(-rmat.T, tvec)
    ax.scatter(C[0], C[1], C[2], s=10, marker="s")
    ax.quiver(C[0], C[1], C[2], corner_rays[0,:], corner_rays[1,:], corner_rays[2,:], color=corner_colors)

    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    return tvec, rmat

#points: nx2 np.float32 array
#mtx: camera matrix
#dist: distortion values
#rays: nx1x3 np.float32 array
def pixel2ray(points, mtx, dist):
    undist_points = cv2.undistortPoints(points, mtx, dist)
    rays = cv2.convertPointsToHomogeneous(undist_points)
    #print("rays: ", rays)
    norm = np.sum(rays**2, axis = -1)**.5
    #print("norm: ", norm)
    rays = rays/norm.reshape(-1, 1, 1)
    return rays


#code for setting axes to be equal in matplotlib3D
#taken from https://stackoverflow.com/questions/13685386/63625222#63625222

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
