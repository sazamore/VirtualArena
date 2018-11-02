import Leap,  cv2, os
import numpy as np
import ImageCorrection as IC
import ctypes 
import pdb

#TODO: Rescale image (about 2x as big)

#os.chdir('/home/virtual_arena/TestTracks')
#start leap daemon (sudo leapd in terminal) before starting working controller

def setCtrlr():
	controller = Leap.Controller()
	controller.set_policy(Leap.Controller.POLICY_IMAGES) #free images for external use
	frame = controller.frame()
	images = frame.images

	#left_image = images[0]
	#right_image = images[1]
	return controller,frame 

def fmtImg(image):
	image_buffer_ptr = image.data_pointer
	ctype_array_def = ctypes.c_ubyte * image.width * image.height

	# as ctypes array
	as_ctype_array = ctype_array_def.from_address(int(image_buffer_ptr))
	# as numpy array
	np_image = np.ctypeslib.as_array(as_ctype_array)	
	return np_image

#TODO: make this a class!
def dist((x1,y1),(x2,y2)):
        return np.sqrt((y2-y1)**2+(x2-x1)**2)
def mid((x1,y1),(x2,y2)):
	return x1+((x2-x1)/2), y1+((y2-y1)/2)


def detect(C):
	# initialize the shape name and approximate the contour
        shape = "unidentified"
	peri = cv2.arcLength(C, True)
        approx = cv2.approxPolyDP(C, 0.04 * peri, True)

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        if len(approx) == 4:
                # compute the bounding box of the contour and use the
        	# bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
		shape = "rectangle"
		return shape, approx
	# if the shape is a pentagon, it will have 5 vertices
	elif len(approx) == 5:
		shape = "pentagon"
 		return shape,approx
	elif len(approx) ==6:
		shape = "hexagon"
		return shape,approx
	# otherwise, we assume the shape is a circle
#	else:
#		shape = "circle"
	return shape

def fitVect(approx,(cX,cY)):
	#TODO: get approx to be output from detect method. Will be faster.
	#peri = cv2.arcLength(C, True)
        #approx = cv2.approxPolyDP(C, 0.04 * peri, True) 
	if isinstance(approx,basestring):
		#print ("You've passed a string! How strange!")
		return ([],[])
	else:
		dists=[]
		numBin = np.array(len(approx))-1

		for i in range(numBin):
			dists.append(dist(approx[i][0],approx[i+1][0]))
		dists.append(dist(approx[numBin][0],approx[0,0]))

		smallest = list(np.where(dists==np.min(dists))[0])
		
		#TODO: clean this up
		if len(smallest) ==1 and smallest<numBin and smallest != 5:
			smallest = int(np.array(smallest))
			#try:
			pt1 = mid(approx[smallest][0],approx[smallest+1][0])
			#except ValueError:
				#pt1 = mid(approx[smallest][0][0],approx[smallest+1][0][0])
		elif len(smallest)==1 and  smallest==numBin:
			smallest = int(np.array(smallest))
			pt1 = mid(approx[smallest-1][0],approx[0][0])
		else:
			smallest = int(np.array(smallest[0]))
			#pdb.set_trace()
			pt1 = mid(approx[smallest][0],approx[smallest+1][0])
		if len(approx)==4 or len(approx) == 6:
			#shape is rectangular, assume smallest side is front
#			if smallest == 0:
#				pt2 = mid(approx[2][0],approx[3][0])
#			elif smallest == 1:
#				pt2 = mid(approx[0][0],approx[3][0])
#			elif smallest == 2:
#				pt2 = mid(approx[0][0],approx[1][0])
#			elif smallest == 3:
#				pt2 = mid(approx[1][0],approx[2][0])
			return pt1

#                if len(approx)==6:
                        #shape is pentagon, assume smallest side is front
#			if smallest < 2:
#        	        	pt2 = mid(approx[smallest+3][0],approx[smallest+4][0])
#                	elif smallest ==2:
#				pt2 = mid(approx[5][0],approx[0][0])
#			else:
#                                pt2 = mid(approx[smallest-3][0],approx[smallest-2][0])
#			return pt1,pt2
			
		if len(approx)==5:
			return ([],[])

def fitCont(threshImg):
        #enter thresholded image
	img = threshImg.copy()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_, cnts,_= cv2.findContours(img,cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

        for C in cnts:
                #get center of contour, find name of shape
                M = cv2.moments(C)
		try:
	                cX = int(M["m10"]/M["m00"])
	                cY = int(M["m01"]/M["m00"])
                except ZeroDivisionError:
			cX = 0
			cY = 0
		shape = detect(C)
		
                #draw contour
                cv2.drawContours(threshImg, [C],-1, (0,255,0),1)

		if len(shape)>1:  # len(shape[1])==4 or len(shape[1]==5):
#			cv2.putText(threshImg, shape[0], (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
#				0.5, (255, 255, 255), 2)
			#cv2.circle(threshImg,(cX,cY),2,(0,0,255),-1)
			#print shape[0]
			pt1 = fitVect(shape[1],(cX,cY))
			#print pt1
			try:
				cv2.arrowedLine(threshImg,(cX,cY),pt1,(255,0,0),1)
			except TypeError:
				continue
	if np.any(pt1==[]):
		return [([],[]),([],[])]
	else:
		return [(cX,cY),pt1]

def process(controller,draw=True, save=False,x=240.,y=640,fps=30,thresh=150):
	print('Press the q key to interrupt')
	check = False
	prefix = []

	Rvect=[]
	Lvect=[]
	if save:
		prefix = raw_input("Enter prefix for video file. ")
		Rname = prefix+'-R.avi'
		Lname = prefix+'-L.avi'
		Rfiltname = prefix + '-tR.avi'
		Lfiltname = prefix + 'tL.avi'
		fourcc= cv2.VideoWriter_fourcc('M','P','E','G') 
		Lmov = cv2.VideoWriter(Lname,
				fourcc,fps,(640,240)) #,apiPreference=1900)
        	Rmov = cv2.VideoWriter(Rname,
                                fourcc,fps,(640,240))
		LmovFilt = cv2.VideoWriter(Lfiltname,
                                fourcc,fps,(640,240))
		RmovFilt = cv2.VideoWriter(Rfiltname,
                                fourcc,fps,(640,240))
	#board_dims = tuple((6,4))
	while (True):
		frame = controller.frame()
		images = frame.images
		if images[0].is_valid and images[1].is_valid:
			left = fmtImg(images[0])
			right = fmtImg(images[1])

		        #found, corners = cv2.findChessboardCorners(left, board_dims,cv2.CALIB_CB_FAST_CHECK,cv2.CALIB_CB_ADAPTIVE_THRESH)
			#found, gridpts = cv2.findCirclesGrid(left,board_dims, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
			#if found:
			#	chess=cv2.drawChessboardCorners(left,board_dims, corners, found)
			#	cv2.imshow('chess',chess)
			#	cv2.waitKey(10)
			#	pdb.set_trace()

			left = cv2.cvtColor(left,cv2.COLOR_GRAY2BGR)
			right = cv2.cvtColor(right,cv2.COLOR_GRAY2BGR)

			_, t_left = cv2.threshold(left,thresh,255,cv2.THRESH_BINARY)
                        _, t_right = cv2.threshold(right,thresh,255,cv2.THRESH_BINARY)

			Rvect.append(fitCont(t_right))
			Lvect.append(fitCont(t_left))
			if save:
				leftname = prefix + '-' +  np.str(frame.id) + '-L.tiff'
				rightname = prefix + '-' +  np.str(frame.id) + '-R.tiff'
				
				Lmov.write(left)
                                Rmov.write(right)
				LmovFilt.write(t_left)
				RmovFilt.write(t_right)
			if draw:

				height, width = left.shape[:2]
                                #left = cv2.resize(left,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
                                #right = cv2.resize(right,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
                                #t_left = cv2.resize(t_left,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				#t_right = cv2.resize(t_right,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
				cv2.imshow('left',left)
				cv2.imshow('right',right)
				cv2.imshow('leftThresh',t_left)
				cv2.imshow('rightThresh',t_right)
				cv2.moveWindow('right',680,240)
				cv2.moveWindow('left',0,240)
				cv2.moveWindow('rightThresh',680,0)
				cv2.moveWindow('leftThresh',0,0)
			#Press the q key to quit
			if cv2.waitKey(1) & 0XFF == ord('q'):
				if save:
					Lmov.release()
					Rmov.release()
					LmovFilt.release()
					RmovFilt.release()
				cv2.destroyAllWindows()
				print(os.getcwd())
				break
		else:
			print('Frames not grabbed. Have you started the leap daemon?')
	return Rvect,Lvect

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

def undistort(image, coordinate_map, coefficient_map, width, height):
    destination = np.empty((width, height), dtype = np.ubyte)

    #wrap image data in numpy array
    i_address = int(image.data_pointer)
    ctype_array_def = ctypes.c_ubyte * image.height * image.width
    # as ctypes array
    as_ctype_array = ctype_array_def.from_address(i_address)
    # as numpy array
    as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
    img = np.reshape(as_numpy_array, (image.height, image.width))

    #remap image to destination
    destination = cv2.remap(img,
                            coordinate_map,
                            coefficient_map,
                            interpolation = cv2.INTER_LINEAR)

    #resize output to desired destination size
    destination = cv2.resize(destination,
                             (width, height),
                             0, 0,
                             cv2.INTER_LINEAR)
    return destination

def runUndist(Controller):
	maps_initialized = False
	while True:
		frame= controller.frame()
		image = frame.images[0]
		if image.is_valid:
			if not maps_initialized:
				left_coordinates, left_coefficients = convert_distortion_maps(frame.images[0])
				right_coordinates, right_coefficients = convert_distortion_maps(frame.images[1])
				maps_initialized = True

		undistorted_left = undistort(image, left_coordinates, left_coefficients, 400, 400)
                undistorted_right = undistort(image, right_coordinates, right_coefficients, 400, 400)

                #display images
                cv2.imshow('Left Camera', undistorted_left)
                cv2.imshow('Right Camera', undistorted_right)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
