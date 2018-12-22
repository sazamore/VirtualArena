import Leap, cv2, os
import numpy as np
import ImageCorrection as IC
import ctypes, yaml
import pdb
import LeapDrawImage as di

"""Use this script to grab chess boards and other preliminary work to calibrate cameras"""

controller, _ = di.setCtrlr()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(controller,board_dims,iter=0):
	# Iter determines how many frames to grab. 0 is 1 frame, any 
	#other value is until user quits
	# Arrays to store object points and image points from all the images.
	#objpoints = [] # 3d point in real world space
	#imgpoints = [] # 2d points in image plane.

	# termination criteria
	#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	if not type(board_dims) == tuple:
		board_dims = (board_dims[0],board_dims[1])

	objp = np.zeros((board_dims[0]*board_dims[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:board_dims[1],0:board_dims[0]].T.reshape(-1,2)

	while (True):
		frame = controller.frame()
		images = frame.images

                if images[0].is_valid and images[1].is_valid:
                	left = di.fmtImg(images[0])
                        right = di.fmtImg(images[1])

                        #Lfound, Lcorners = cv2.findChessboardCorners(left, board_dims, cv2.CALIB_CB_ADAPTIVE_THRESH)
			Rfound, Rcorners = cv2.findChessboardCorners(right, board_dims, cv2.CALIB_CB_ADAPTIVE_THRESH)
                        #Rfound, Rcorners = cv2.findCirclesGrid(right, board_dims)
			if Rfound:
                                Rchess=cv2.drawChessboardCorners(right,board_dims, Rcorners, Rfound)
                                cv2.imshow('chess-R',Rchess)
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(right, Rcorners,(11,11),(-1,-1), criteria)
				imgpoints.append(corners2)
				if iter == 0:
					cv2.imwrite('rawL',left)
					cv2.imwrite('rawR',right)
					cv2.destroyAllWindows()
					break
                        #        pdb.set_trace()
			else:
				cv2.imshow('chess-R',right)
			#if Lfound:
                        #        Lchess=cv2.drawChessboardCorners(left,board_dims, Lcorners, Lfound)
                        #        cv2.imshow('chess-L',Lchess)
			#	 pdb.set_trace()
			#else:
			#	cv2.imshow('chess-L',left)

                        if cv2.waitKey(1) & 0XFF == ord('q'):
				cv2.destroyAllWindows()
				break
	print('Calibrating camera. This may take a few minutes.')
	ret,mtx,dist,rvects,tvects = cv2.calibrateCamera(objpoints,imgpoints,(640,240),None,None)

	return objpoints, imgpoints, [ret,mtx,dist,rvects,tvects]

def convert_distortion_maps(image):

    distortion_length = image.distortion_width * image.distortion_height
    xmap = np.zeros(distortion_length/2, dtype=np.float32)
    ymap = np.zeros(distortion_length/2, dtype=np.float32)

    for i in range(0, distortion_length, 2):
        xmap[distortion_length/2 - i/2 - 1] = image.distortion[i] * image.width
        ymap[distortion_length/2 - i/2 - 1] = image.distortion[i + 1] * image.height

    xmap = np.reshape(xmap, (image.distortion_height, image.distortion_width/2))
    ymap = np.reshape(ymap, (image.distortion_height, image.distortion_width/2))

    #resize the distortion map to equal desired destination image size
    resized_xmap = cv2.resize(xmap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)
    resized_ymap = cv2.resize(ymap,
                              (image.width, image.height),
                              0, 0,
                              cv2.INTER_LINEAR)

    #Use faster fixed point maps
    coordinate_map, interpolation_coefficients = cv2.convertMaps(resized_xmap,
                                                                 resized_ymap,
                                                                 cv2.CV_32FC1,
                                                                 nninterpolation = False)

    return coordinate_map, interpolation_coefficients


def undistort(img, mtx,dist):
	h,w = img.shape[:2]
	newcammtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
	dst=cv2.undistort(img,mtx,dist,None,newcammtx)
	x,y,h,w = roi
	#dst = dst[y:h,x:x+w] #resize image
	cv2.imwrite('calibadjust.png',dst)
	return newcammtx, roi

def remap(img, roi, mtx, dist,rvecs,tvecs,newcammtx):
	# undistort
	w,h = img.shape[:2]
	mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcammtx,(w,h),5)
	dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

	# crop the image
	x,y,w,h = roi
	dst = dst[y:y+h, x:x+w]
	cv2.imwrite('calibresult.png',dst)
	cv2.waitKey(0)

	#tot_error = 0
	#for i in xrange(len(objpoints)):
	#	imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
	#	error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
	#	tot_error += error

	#print "total error: ", tot_error,len(objpoints)
	return mapx,mapy

def store(name,mtx,dist):
	"""Stores matrix, and distances in a JSON file"""
	with open(name,"w") as f:
		yaml.dump(mtx,f)
		yaml.dump(dist,f)
	print("Save complete.")

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,255,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255,255,255), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (127,127,127), 5)
    return img

def drawCube(img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        #draw ground floor in gray
        img = cv2.drawContours(img, [imgpts[:4]],-1,(127,127,127),-3)

        #draw wire frame in white
        for i,j in zip(range(4),range(4,8)):
                img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
	# draw top layer in black
	img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,0),3)
	return img

def grabBoard(controller,iter=1,save=False):
	"Plots images until chessboard is found. Then returns images and exits"
	count=0
	fps=30
	if save:
		os.chdir('/home/virtual_arena/calIMG/')
		prefix = raw_input("Enter prefix for video file. ")
		Rname = prefix+'-R.avi'
		Lname = prefix+'-L.avi'
		fourcc= cv2.VideoWriter_fourcc('M','P','E','G') 
		Lmov = cv2.VideoWriter(Lname,
				fourcc,fps,(640,240)) #,apiPreference=1900)
		Rmov = cv2.VideoWriter(Rname,
				fourcc,fps,(640,240)) #,apiPreference=1900)
        
	while (True):
		frame = controller.frame()
		images = frame.images
		left = di.fmtImg(images[0])
		right = di.fmtImg(images[1])
		rfound = cv2.findChessboardCorners(right,(6,4),cv2.CALIB_CB_ADAPTIVE_THRESH)
		#lfound = cv2.findChessboardCorners(left,(6,4),cv2.CALIB_CB_ADAPTIVE_THRESH)

		if save:
				mleft = cv2.cvtColor(left,cv2.COLOR_GRAY2BGR)
				mright = cv2.cvtColor(right,cv2.COLOR_GRAY2BGR)
				
				Lmov.write(mleft)
				Rmov.write(mright)
        
		if rfound[0]:
			orig = os.getcwd()
			os.chdir('/home/virtual_arena/calIMG/')
                        cv2.imwrite('LEFTchess-'+np.str(count)+'.png',left)
                        cv2.imwrite('RIGHTchess-'+np.str(count)+'.png',right)
                        Rchess=cv2.drawChessboardCorners(right,(6,4), rfound[1], rfound[0])

			lfound = cv2.findChessboardCorners(left,(6,4),cv2.CALIB_CB_ADAPTIVE_THRESH)
			cv2.imshow('right',right)
			if lfound[0]:
				Lchess=cv2.drawChessboardCorners(left,(6,4), lfound[1], lfound[0])
				cv2.imshow('left',left)
				#cv2.waitKey(0)
				#cv2.imwrite('chessL.png',left)
				#cv2.imwrite('chessR.png',right)
				count += 1
				print('Board grabbed successfully.')

				if count==iter:
					if save:
						Lmov.release()
						Rmov.release()
					print('Board grab complete!, closing...')#return left, right
					cv2.destroyAllWindows()
					break
		else:
			cv2.imshow('left',left)

		if cv2.waitKey(1) & 0XFF == ord('q'):
			os.chdir(orig)
			cv2.destroyAllWindows()
			break

def testCal(controller,board_dims,mtx, dist):
	objp = np.zeros((board_dims[0]*board_dims[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:board_dims[1],0:board_dims[0]].T.reshape(-1,2)

	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
	axisCube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
        	           [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

 	#calibrate
        #ret,mtx,dist,rvects,tvects = cv2.calibrateCamera(objpoints,imgpoints,(640,240),None,None)

        while (True):
                frame = controller.frame()
                images = frame.images

                if images[0].is_valid and images[1].is_valid:
                        left = di.fmtImg(images[0])
                        right = di.fmtImg(images[1])

                        Lfound, Lcorners = cv2.findChessboardCorners(left, board_dims, cv2.CALIB_CB_ADAPTIVE_THRESH)
                        Rfound, Rcorners = cv2.findChessboardCorners(right, board_dims, cv2.CALIB_CB_ADAPTIVE_THRESH)
                        if Rfound:
				corners2 = cv2.cornerSubPix(right,Rcorners,(11,11),(-1,-1),criteria)

				#Find the rotation and translation vectors.
				ret,rvecs,tvecs = cv2.solvePnP(objp,corners2,mtx,dist)

				# project 3D points to image plane
				#imgpts, jac = cv2.projectPoints(axis,rvecs,tvecs,mtx,dist)
				#img = draw(right, corners2, imgpts)

				imgpts, jac = cv2.projectPoints(axisCube, rvecs,tvecs,mtx,dist)
				img = drawCube(right,corners2,imgpts)
				#SOMETHING IS WRONG WITH DRAWCUBE

				cv2.imshow('right',img)
				#k = cv2.waitKey(0) & 0xFF
			else:
				cv2.imshow('right',right)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
