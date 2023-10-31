import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template,Response
from werkzeug.utils import secure_filename
import ffmpeg
global globalFilename

# from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import cv2
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter
# import seaborn as sns
import os.path
import time
import pandas as pd
from filterpy.kalman import KalmanFilter
import os
from PIL import Image
from pathlib import Path
import csv
import seaborn as sns
camera = cv2.VideoCapture(0)

def generate_frames():
	while True:
		sucess,frame = camera.read()
		if not sucess:
			break
		else:
			ret,buffer = cv2.imencode('.jpg',frame)
			frame=buffer.tobytes()
		yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def preprocess_data(filename):
	folder_id = 8

	# base = url_for("static", filename = "Data/" + str(folder_id) + "/")

	# base = "E:/JonathanSir/New Data/Data/" + str(folder_id) + "/"
	base = app.static_folder+"/Data/"+str(folder_id) +"/"
	# print(base)
	# cap = cv2.VideoCapture(r"static/uploads/" + filename)
	cap = cv2.VideoCapture(os.path.join(app.static_folder, "uploads/", filename))
	print(os.path.join(app.static_folder, "uploads/", filename))
	# base = "E:/JonathanSir/New Data/Data/" + str(folder_id) + "/"
	# cap = cv2.VideoCapture(r"E:/JonathanSir/New Data/P3_enhanced.mp4")

	Path(base + 'binary_segmented').mkdir(parents=True, exist_ok=True)
	Path(base+'colour_segmented').mkdir(parents=True, exist_ok=True)
	Path(base + 'original').mkdir(parents=True, exist_ok=True)
	Path(base + 'det').mkdir(parents=True, exist_ok=True)


	print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
	fno = 1
	ret = True
	fps = cap.get(cv2.CAP_PROP_FPS)
	hourval = 0
	minuteval = 0.00
	print("Processing Initial Frames")
	# flash("Processing Initial Frames")
	with open(base + "det/det.txt", "w", newline='') as o:

		while True:
			###########Time Stamp Calculation#########
			
			print("Frame: ",fno,"Frame status: ",ret,"time: ","{:.2f}".format(float(hourval+minuteval)))
			# flash("Frame: ",fno,"Frame status: ",ret,"time: ","{:.2f}".format(float(hourval+minuteval)))
			minuteval +=0.05
			if minuteval == 0.60:
				hourval+=1
				minuteval = 0.00
			# print("Frame: ",fno,"Frame status: ",ret,"time: ",str(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))
			ret, image = cap.read()  # ret is a boolean variable that returns true if the frame is available.
			# height, width, channels = image.shape
			if not ret:
				break
			# x = 0
			# w =1024
			# y = 0
			# h = 1024
			#for bac v3
			# y = 338
			# h = 850

			# x = 775
			# w = 1287
			#org_img = frame[290:700,550:1300]
			#for bac v3
			# org_img = frame[y:h,x:w]
			cv2.namedWindow("Selection Window", cv2.WINDOW_NORMAL)
			cv2.resizeWindow("Selection Window", 1920, 1080)
			if (fno ==1):
				r = cv2.selectROI("Selection Window",image,False,False)
			org_img = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
			#for bacv3
			ret1, mask = cv2.threshold(org_img, 115, 255, cv2.THRESH_BINARY_INV)
        	#for bacv4
        	#ret1, mask = cv2.threshold(org_img, 110, 255, cv2.THRESH_BINARY_INV)
			mask = mask[:,:,0]
			ret2, mask1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

			contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
			for c in contours:
				# get the bounding rect
				x, y, w, h = cv2.boundingRect(c)
				if (w * h < 5000):
					# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
					# data=[fno,-1,x,y,x+w,y+h,1,-1,-1,-1]
					data = [fno, -1, x, y, w, h, 1, -1, -1, -1]
					writer = csv.writer(o, delimiter=',')
					writer.writerow(data)
			cv2.imwrite(base + 'original/' + str(fno) + '.png', org_img)
			cv2.imwrite(base + 'binary_segmented/' + str(fno) + '.png', mask1)
			plt.imsave(base + "colour_segmented/" + str(fno) + '.png', mask1)

			fno = fno + 1
			#seq_dets = np.array(seq_dets)
			# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
			cv2.putText(org_img, "Time: "+str("{:.2f}".format(float(hourval+minuteval))),(10,20) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			
			# cv2.imshow("Frame", image)
			# cv2.resizeWindow("Frame", 1920, 1080)
			# cv2.imshow("Selected Region", org_img)
			#cv2.imshow("Mask", mask)
			# print(filename)
			# net_ret,net_buffer = cv2.imencode('jpg',mask)
			# mask = net_buffer.tobytes()
			# key = cv2.waitKey(5)
			# if key == 27: # esc key on keyboard
			#     break
			# else:
			#     break
			mask_ret,org_img_buffer = cv2.imencode('.jpg',org_img)
			org_img=org_img_buffer.tobytes()
			# image_ret,image_buffer = cv2.imencode('.jpg',image)
			# image=image_buffer.tobytes()
		
			yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + org_img + b'\r\n')
			# yield(b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
		cap.release()
		cv2.destroyAllWindows()
def trackCol(filename):
	folder_id = 8
	base = app.static_folder+"/Data/"+str(folder_id) +"/"
	print(base)
	try:
		from numba import jit
	except:
		def jit(func):
			return func

	np.random.seed(0)


	def linear_assignment(cost_matrix):
		try:
			import lap
			_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
			return np.array([[y[i], i] for i in x if i >= 0])  #
		except ImportError:
			from scipy.optimize import linear_sum_assignment
			x, y = linear_sum_assignment(cost_matrix)
			return np.array(list(zip(x, y)))


	@jit
	def iou(bb_test, bb_gt):
		"""
		Computes IUO between two bboxes in the form [x1,y1,x2,y2]
		"""
		xx1 = np.maximum(bb_test[0], bb_gt[0])
		yy1 = np.maximum(bb_test[1], bb_gt[1])
		xx2 = np.minimum(bb_test[2], bb_gt[2])
		yy2 = np.minimum(bb_test[3], bb_gt[3])
		w = np.maximum(0., xx2 - xx1)
		h = np.maximum(0., yy2 - yy1)
		wh = w * h
		o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
				+ (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
		return (o)


	def convert_bbox_to_z(bbox):
		"""
		Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
		[x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
		the aspect ratio
		"""
		w = bbox[2] - bbox[0]
		h = bbox[3] - bbox[1]
		x = bbox[0] + w / 2.
		y = bbox[1] + h / 2.
		s = w * h  # scale is just area
		r = w / float(h)
		return np.array([x, y, s, r]).reshape((4, 1))


	def convert_x_to_bbox(x, score=None):
		"""
		Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
		[x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
		"""
		w = np.sqrt(x[2] * x[3])
		h = x[2] / w
		if (score == None):
			return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
		else:
			return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


	class KalmanBoxTracker(object):
		"""
		This class represents the internal state of individual tracked objects observed as bbox.
		"""
		count = 0

		def __init__(self, bbox):
			"""
			Initialises a tracker using initial bounding box.
			"""
			# define constant velocity model
			self.kf = KalmanFilter(dim_x=7, dim_z=4)
			self.kf.F = np.array(
				[[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
				[0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
			self.kf.H = np.array(
				[[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

			self.kf.R[2:, 2:] *= 10.
			self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
			self.kf.P *= 10.
			self.kf.Q[-1, -1] *= 0.01
			self.kf.Q[4:, 4:] *= 0.01

			self.kf.x[:4] = convert_bbox_to_z(bbox)
			self.time_since_update = 0
			self.id = KalmanBoxTracker.count
			KalmanBoxTracker.count += 1
			self.history = []
			self.hits = 0
			self.hit_streak = 0
			self.age = 0

		def update(self, bbox):
			"""
			Updates the state vector with observed bbox.
			"""
			self.time_since_update = 0
			self.history = []
			self.hits += 1
			self.hit_streak += 1
			self.kf.update(convert_bbox_to_z(bbox))

		def predict(self):
			"""
			Advances the state vector and returns the predicted bounding box estimate.
			"""
			if ((self.kf.x[6] + self.kf.x[2]) <= 0):
				self.kf.x[6] *= 0.0
			self.kf.predict()
			self.age += 1
			if (self.time_since_update > 0):
				self.hit_streak = 0
			self.time_since_update += 1
			self.history.append(convert_x_to_bbox(self.kf.x))
			return self.history[-1]

		def get_state(self):
			"""
			Returns the current bounding box estimate.
			"""
			return convert_x_to_bbox(self.kf.x)


	def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
		"""
		Assigns detections to tracked object (both represented as bounding boxes)

		Returns 3 lists of matches, unmatched_detections and unmatched_trackers
		"""
		if (len(trackers) == 0):
			return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
		iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

		for d, det in enumerate(detections):
			for t, trk in enumerate(trackers):
				iou_matrix[d, t] = iou(det, trk)

		if min(iou_matrix.shape) > 0:
			a = (iou_matrix > iou_threshold).astype(np.int32)
			if a.sum(1).max() == 1 and a.sum(0).max() == 1:
				matched_indices = np.stack(np.where(a), axis=1)
			else:
				matched_indices = linear_assignment(-iou_matrix)
		else:
			matched_indices = np.empty(shape=(0, 2))

		unmatched_detections = []
		for d, det in enumerate(detections):
			if (d not in matched_indices[:, 0]):
				unmatched_detections.append(d)
		unmatched_trackers = []
		for t, trk in enumerate(trackers):
			if (t not in matched_indices[:, 1]):
				unmatched_trackers.append(t)

		# filter out matched with low IOU
		matches = []
		for m in matched_indices:
			if (iou_matrix[m[0], m[1]] < iou_threshold):
				unmatched_detections.append(m[0])
				unmatched_trackers.append(m[1])
			else:
				matches.append(m.reshape(1, 2))
		if (len(matches) == 0):
			matches = np.empty((0, 2), dtype=int)
		else:
			matches = np.concatenate(matches, axis=0)

		return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


	class Sort(object):
		def __init__(self, max_age=80, min_hits=5):
			"""
			Sets key parameters for SORT
			"""
			self.max_age = max_age
			self.min_hits = min_hits
			self.trackers = []
			self.frame_count = 0

		def update(self, dets=np.empty((0, 5))):
			"""
			Params:
			dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
			Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
			Returns the a similar array, where the last column is the object ID.

			NOTE: The number of objects returned may differ from the number of detections provided.
			"""
			self.frame_count += 1
			# get predicted locations from existing trackers.
			trks = np.zeros((len(self.trackers), 5))
			to_del = []
			ret = []
			for t, trk in enumerate(trks):
				pos = self.trackers[t].predict()[0]
				trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
				if np.any(np.isnan(pos)):
					to_del.append(t)
			trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
			for t in reversed(to_del):
				self.trackers.pop(t)
			matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

			# update matched trackers with assigned detections
			for m in matched:
				self.trackers[m[1]].update(dets[m[0], :])

			# create and initialise new trackers for unmatched detections
			for i in unmatched_dets:
				trk = KalmanBoxTracker(dets[i, :])
				self.trackers.append(trk)
			i = len(self.trackers)
			for trk in reversed(self.trackers):
				d = trk.get_state()[0]
				if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
					ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
				i -= 1
				# remove dead tracklet
				if (trk.time_since_update > self.max_age):
					self.trackers.pop(i)
			if (len(ret) > 0):
				return np.concatenate(ret)
			return np.empty((0, 5))


	def parse_args():
		"""Parse input arguments."""
		parser = argparse.ArgumentParser(description='SORT demo')
		parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
							action='store_true')
		parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data5')
		parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
		args = parser.parse_args()
		return args


	#rad = [[] for _ in range(100)]
	rad = np.zeros([1000, 1000], dtype = float)
	areaBac = np.zeros([1000, 1000], dtype = float)
	realImageHeight = 1000
	realImageWidth = 1000
	lengthPerPixelsWidth = realImageWidth/1920
	lengthPerPixelsHeight = realImageHeight/1080
	# print(fps)
	minuteval = 0.00
	hourval = 0
	# if __name__ == '__main__':

	# all train
	idList = []
	timeStamp = []
	args = parse_args()
	display = args.display
	phase = args.phase
	total_time = 0.0
	total_frames = 0
	colours = np.random.rand(32, 3)  # used only for display
	########### For Visualization#################
	# if (display==False):
	# 	plt.ion()
	# 	fig = plt.figure()
	# 	ax1 = fig.add_subplot(111, aspect='equal')
	##############################################

	if not os.path.exists(base + 'output'):
		os.makedirs(base + 'output')
	if not os.path.exists(base + 'track'):
		os.makedirs(base + 'track')
	pattern = os.path.join(base, 'det', 'det.txt')
	print(pattern)
	for seq_dets_fn in glob.glob(base + '/det/det.txt'):
		print('Tracking....')
		mot_tracker = Sort()  # create instance of the SORT tracker
		seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
		seq = seq_dets_fn[pattern.find('*'):].split('\\')[0]
		print(seq)
		with open(base + 'output/%s.txt' % (seq), 'w') as out_file:
			print("Processing %s." % (seq))
			for frame in range(int(seq_dets[:, 0].max())):
				frame += 1  # detection and frame numbers begin at 1
				dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
				dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
				total_frames += 1
				if (display==False):
					fn = base + 'original/%d.png' % (frame)
					im = io.imread(fn)
					im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
					fn1 = base + 'binary_segmented/%d.png' % (frame)
					mask = io.imread(fn1, 0)
					ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
					contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
					cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
					# draw group
					fn_group = base + 'colour_segmented/%d.png' % (frame)
					light_yellow = (255, 0, 0)
					dark_yellow = (255, 255, 0)
					img3 = Image.open(fn_group)
					pixels = img3.load()  # this is not a list, nor is it list()'able
					width, height = img3.size
					for i in range(width):
						for j in range(height):
							if (pixels[i, j] != (255, 255, 0)):
								pixels[i, j] = (0, 0, 0)
					opencvImage = cv2.cvtColor(np.array(img3), cv2.COLOR_RGB2BGR)
					opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
					ret, thresh1 = cv2.threshold(opencvImage, 127, 255, cv2.THRESH_BINARY)
					contours_group, hier = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

					for cnt in contours_group:
						area = cv2.contourArea(cnt)
						if (area > 2):
							M = cv2.moments(cnt)
							cX = int(M["m10"] / M["m00"])
							cY = int(M["m01"] / M["m00"])
							for cnt1 in contours:
								dist = cv2.pointPolygonTest(cnt1, (cX, cY), True)
								if (dist > 0):
									cv2.drawContours(im, [cnt1], 0, (255, 255, 0), 1)

					# ax1.imshow(im)
					# plt.title(seq + ' Tracked Targets')

				start_time = time.time()
				trackers = mot_tracker.update(dets)
				cycle_time = time.time() - start_time
				total_time += cycle_time

				for d in trackers:
					if d[4] not in idList:
						idList.append(int(d[4]))
					print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
						file=out_file)
					# print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]))
					# print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2]-d[0],d[3]-d[1]),file=out_file)
					# print('%d,%.2f,%.2f,%2f' % (d[4],d[2]-d[0],d[3]-d[1],((d[2]-d[0])+(d[3]-d[1]))/4),file=out_file)
					#rad[int(d[4] - 1)][frame].insert(float((d[2] - d[0]) + (d[3] - d[1]) / 4))
					# d[3] - d[1] == ractangle height
					# d[2] - d[0] == ractangle width
					rad[frame][int(d[4])] = float(((d[2] - d[0]) + (d[3] - d[1])) / 4)#frame wise radiusgrowth data
					areaBac[frame][int(d[4])] = 3.141592653589793238462643383279502884197*float((((d[2] - d[0])*lengthPerPixelsWidth + (d[3] - d[1])*lengthPerPixelsHeight) / 4)**2)#frame wise area growth data
					# rad[float(frame)/fps][int(d[4] - 1)] = float((d[2] - d[0]) + (d[3] - d[1]) / 4)#time wise growth data

					'''
					if int((d[2]-d[0])+(d[3]-d[1])/4) > 5 :
						rad[int(d[4]-1)].append(float((d[2]-d[0])+(d[3]-d[1])/4))

					else:
						rad[int(d[4]-1)].append(0)
					'''
					'''
					if d[4]==1:
						i1.append(((d[2]-d[0])+(d[3]-d[1]))/4)
					elif d[4]==2:
						i2.append(((d[2]-d[0])+(d[3]-d[1]))/4)
					elif d[4]==3:
						i3.append(((d[2]-d[0])+(d[3]-d[1]))/4)
					elif d[4]==4:
						i4.append(((d[2]-d[0])+(d[3]-d[1]))/4)
					elif d[4]==5:
						i5.append(((d[2]-d[0])+(d[3]-d[1]))/4)
						#data=[(d[2]+d[3])/4]

					'''
					"""
					data=[((d[2]-d[0])+(d[3]-d[1]))/4]
					writer = csv.writer(out_file,delimiter=' ')
					writer.writerow(data)
					"""
					x = float((d[2] - d[0]) + (d[3] - d[1]) / 4)



					if (display==False):
						d = d.astype(np.int32)
						# ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=2,ec=colours[d[4]%32,:]))
						if (d[4] == 9):
							cv2.putText(im, str(d[4]),(d[0], d[1]) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)#Id
							cv2.putText(im, str(x), (d[0], d[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))#rad
							# ax1.text(d[0], d[1], str(d[4]), color=[1, 0, 0], fontsize=15)
						else:
							cv2.putText(im, str(d[4]), (d[0], d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
							cv2.putText(im, str(x), (d[0], d[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
							# ax1.text(d[0], d[1], str(d[4]), color=[0, 1, 0], fontsize=15)
				cv2.imwrite(base + 'track/' + str(frame) + '.png', im)
				#collecting time Stamp
				# timeStamp.append(float(frame)/fps)
				timeStamp.append("{:.2f}".format(float(hourval+minuteval)))
				# cv2.putText(im, "Time: "+str(float(frame)/fps),(10,20) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
				# Time stamp calculation

				

				cv2.putText(im, "Time: "+str("{:.2f}".format(float(hourval+minuteval))),(10,20) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
				minuteval +=0.05
				if minuteval == 0.60:
					hourval+=1
					minuteval = 0.00
				#######################
				# cv2.imshow("Frame", im)
				im_ret,im_buffer = cv2.imencode('.jpg',im)
				im=im_buffer.tobytes()
				# image_ret,image_buffer = cv2.imencode('.jpg',image)
				# image=image_buffer.tobytes()
		
				yield(b'--frame\r\n'
                  	 b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')

			#################Optinal###########
				# if (display==False):
				# 	fig.canvas.flush_events()
				# 	plt.draw()
				# 	ax1.cla()


			###############################
		# df = pd.DataFrame(list(zip(*[i2, i3, i4]))).add_prefix('ID')
		#df = pd.DataFrame(rad)
		#df.to_csv(base + 'Radius.csv', index=False)s
		"""
		writer = csv.writer(open(base + 'output/t.csv', 'a+', newline =''))
		writer.writerow(i2)
		writer.writerow(i3)
		writer.writerow(i4)
		"""

	print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
		total_time, total_frames, total_frames / total_time))

	if (display==False):
		print("Note: to get real runtime results run without the option: --display")
	#key = cv2.waitKey(10)
	cv2.destroyAllWindows()
	plt.close()
	print("Tracking Successful")
	dfRadius = pd.DataFrame(rad).add_prefix('ID_')
	#df.add_prefix('ID_')
	dfRadius.to_csv(base + 'Radius.csv', index=False)
	dfarea = pd.DataFrame(areaBac).add_prefix('ID_')
	#df.add_prefix('ID_')
	dfarea.to_csv(base + 'Area.csv', index=False)
	i=0
	img_array = []
	for filename in glob.glob(base + "track/*.png"):
		# print(filename)
		# print(i)
		img = cv2.imread(base + f"track/{i+1}.png")
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)
		i+=1
	
	
	out = cv2.VideoWriter(base + "test.avi",cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
	
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()
	print(idList)
	print(timeStamp)

	#Graph Analysis Part
	# radiusVal = np.zeros([len(idList), len(timeStamp)], dtype = float)

	# for j in idList:
	#     for i in range(len(timeStamp)):
	#         print(rad[i][j])
	x_data = pd.DataFrame({'Time' : timeStamp})

	dfRadius = pd.read_csv(base + "Radius.csv")
	dfArea = pd.read_csv(base + "Area.csv")
	plotDataRadius = x_data
	plotDataArea = x_data
	dfmRadiusWithTime = pd.concat([x_data,dfRadius],axis=1)
	for i in idList:
		plotDataRadius = pd.concat([plotDataRadius,dfRadius["ID_" +str(i)][:len(timeStamp)]],axis=1)     
		plotDataArea = pd.concat([plotDataArea,dfArea["ID_" +str(i)][:len(timeStamp)]],axis=1)     
		#print(df["ID_" +str(i)])
	dfmRadius = plotDataRadius.melt('Time', var_name='cols', value_name='vals')
	dfmRadiusWithTime.to_csv(base + "RadiusWithTimeStamp.csv", index = False)
	dfmRadius.to_csv(base + "SortedIdRadiusAnalysis.csv", index = False)
	dfmArea = plotDataArea.melt('Time', var_name='cols', value_name='vals')
	dfmArea.to_csv(base + "SortedIdAreaAnalysis.csv", index = False)
	###########Visualization Part#########################
	

	# plt.figure() 
	# plt.show()
	# x_data = timeStamp
	print("Track Sucessfull")

def graph_generate():
	print("In Graph")
	# data = pd.read_csv(url_for('static', filename='Data/8/SortedIdAreaAnalysis.csv'))
	data = pd.read_csv("static/Data/8/SortedIdAreaAnalysis.csv")
	sns.set(rc = {'figure.figsize':(12,8)})
	# sns_plot = sns.lineplot(x="Time", y="vals", hue='cols', data=dfm)
	# sns.boxplot(x="Time", y="vals", hue='cols', data=dfm)
	sns.lineplot(x="Time", y="vals", hue='cols', data=data).figure.savefig("SortedIdRadiusAnalysis.png") 


@app.route('/')
def view():
    return render_template('index_1.html')

@app.route('/home')
def home():
    return render_template('index_1.html')

@app.route('/proceed')
def proceed():
	return render_template('upload.html')
@app.route('/tutorials')
def tutorials():
	return render_template('tutorials.html')
@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_video filename: ' + filename)
		flash('Video successfully uploaded and displayed below')
		# global globalFilename
		# globalFilename = filename
		return render_template('upload.html', filename=filename)

@app.route('/display/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)
@app.route('/setParameters/<filename>')
def setParameters(filename):
	return render_template('setParameters.html',filename=filename)

@app.route('/details/<filename>')
def details(filename):
	# print('static/uploads/' + filename)
	# ffmpeg.probe('static'+ filename='uploads/' + filename)["streams"]
	# print(url_for('static', filename='uploads/' + filename))
	# print(os.path.join(app.static_folder, filename))
	# print(ffmpeg.probe(os.path.join(app.static_folder, filename))["streams"])
	return render_template('details.html',metadata_text=ffmpeg.probe(url_for('static', filename='uploads/' + filename))["streams"])
@app.route('/analyse/<filename>')
def analyse(filename):
	return render_template('analyse.html',filename=filename)
@app.route('/mainvideo/<filename>')
def mainvideo(filename):
	# track(filename)
	return Response(preprocess_data(filename),mimetype='multipart/x-mixed-replace; boundary=frame')


##Testing for Realtime Camera function
# @app.route('/video')
# def video():
# 	# track(filename)
# 	return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/track/<filename>')
def track(filename):
	return render_template('track.html',filename=filename)
@app.route('/trackvideo/<filename>')
def trackvideo(filename):
	# track(filename)
	return Response(trackCol(filename),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/generateGraph')
def generateGraph():
	# track(filename)
	return Response(graph_generate())
@app.route('/exportData')
def exportData():
	return render_template('exportData.html')
@app.route('/visualization')
def visualization():
	return render_template('visualization.html')
if __name__ == "__main__":
    app.run(debug=True)