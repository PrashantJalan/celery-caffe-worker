"""
This file caters to all the jobs for the web server.

Author - Prashant Jalan
"""

from __future__ import absolute_import
from executable.celery import app

"""
The function takes as input:
1) src_path: Input image, directory, or npy. 
2) socketid: The socket id of the connection.
3) result_path:
NOTE:
1) Its job is to classify the images according to the pre-trained model.
2) ignore_result=True signifies that celery won't pass any result to the backend.
3) log_to_terminal is used to publish messages to the redis server.
4) It is important to import all the modules only inside the function
5) When running with new version of caffe do np.load(MEAN_FILE).mean(1).mean(1)
"""
@app.task(ignore_result=True)
def classifyImages(src_path, socketid, result_path):
	try:
		import caffe, numpy as np, os, glob, time, operator, scipy.io as sio

		#Used to assign labels to the results
		matWNID = sio.loadmat(os.path.join(os.path.dirname(__file__),'WNID.mat'))
		WNID_cells = matWNID['wordsortWNID']

		#Caffe Initialisations
		CAFFE_DIR = os.path.normpath(os.path.join(os.path.dirname(caffe.__file__),"..",".."))
		MODEL_FILE = os.path.join(CAFFE_DIR, 'models/bvlc_reference_caffenet/deploy.prototxt')
		PRETRAINED = os.path.join(CAFFE_DIR, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
		MEAN_FILE = os.path.join(CAFFE_DIR, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
		RAW_SCALE = 255.0
		IMAGE_DIMS = (256, 256)
		CHANNEL_SWAP = (2, 1, 0)

		#Set CPU mode
		caffe.set_mode_cpu()

		# Make classifier.
		classifier = caffe.Classifier(MODEL_FILE, PRETRAINED, image_dims=IMAGE_DIMS, 
		mean=np.load(MEAN_FILE), raw_scale=RAW_SCALE,
		channel_swap=CHANNEL_SWAP)

		# Load numpy array (.npy), directory glob (*), or image file.
		input_file = os.path.abspath(src_path)
		if input_file.endswith('npy'):
			print("Loading file: %s" % input_file)
			inputs = np.load(args.input_file)
		elif os.path.isdir(input_file):
			print("Loading folder: %s" % input_file)
			inputs = [caffe.io.load_image(im_f) for im_f in glob.glob(input_file + '/*')]
		else:
			print("Loading file: %s" % input_file)
			inputs = [caffe.io.load_image(input_file)]

		# Classify.
		print("Classifying %d inputs." % len(inputs))
		start = time.time()
		prediction = classifier.predict(inputs)
		print("Done in %.2f s." % (time.time() - start))

		#Send Results
		if os.path.isdir(input_file):
			results = {}
			count = 0
			for im_f in glob.glob(input_file + '/*'):
				dictionary = {}
				for i, j in enumerate(prediction[count]):
					dictionary[i] = j
				predsorted = sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)
				top5 = predsorted[0:5]
				topresults = []
				for item in top5:
					topresults.append([str(WNID_cells[item, 0][0][0]),str(item[1])])
					results[im_f] = topresults
				count += 1
			print results
		else:
			dictionary = {}
			for i, j in enumerate(prediction[0]):
				dictionary[i] = j

			predsorted = sorted(dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)
			top5 = predsorted[0:5]
			topresults = [] 
			for item in top5:
				topresults.append([str(WNID_cells[item, 0][0][0]),str(item[1])])
			print topresults

	except Exception as e:
		#In case of an error, print the whole error with traceback
		import traceback
		print str(traceback.format_exc()), socketid


"""
The function takes as input:
1) src_path: Input image or directory. 
2) output_path: Path to store the decaf features.
2) socketid: The socket id of the connection.
3) result_path:
NOTE:
1) Its job is to find the decaf features of images according to the pre-trained model.
2) ignore_result=True signifies that celery won't pass any result to the backend.
3) log_to_terminal is used to publish messages to the redis server.
4) It is important to import all the modules only inside the function
5) When running with new version of caffe do np.load(MEAN_FILE).mean(1).mean(1)
"""
@app.task(ignore_result=True)
def decafImages(src_path, output_path, socketid, result_path):
	try:
		import caffe, numpy as np, os, glob, time, operator, scipy.io as sio

		#Caffe Initialisations
		CAFFE_DIR = os.path.normpath(os.path.join(os.path.dirname(caffe.__file__),"..",".."))
		MODEL_FILE = os.path.join(CAFFE_DIR, 'models/bvlc_reference_caffenet/deploy.prototxt')
		PRETRAINED = os.path.join(CAFFE_DIR, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

		#Set CPU mode
		caffe.set_mode_cpu()

		# Make classifier.
		classifier = caffe.Classifier(MODEL_FILE, PRETRAINED)

		#Getting absolute paths
		src_path = os.path.abspath(src_path)
		result_path = os.path.abspath(result_path)

		#Find decaf features and send Results
		if os.path.isdir(src_path):
			for input_file in glob.glob(src_path + '/*'):
				print("Loading file: %s" % input_file)
				input_image = caffe.io.load_image(input_file)
				
				#Finding decaf features
				classifier.predict([input_image])
				blobs = classifier.blobs.items()
				features = blobs[-3][1].data[:,:,0,0]
				features_center = blobs[-3][1].data[4,:,0,0]
				features_center = np.resize(features_center, (1,4096))

				#Saving decaf features
				matfile = {}
				matfile['decaf'] = features
				matfile['decaf_center'] = features_center
				out_file = os.path.join(result_path, os.path.basename(input_file)+'.mat'), matfile
				sio.savemat(out_file)
				print matfile
				print "Decaf features calculated and saved at "+out_file
		else:
			input_file = src_path
			print("Loading file: %s" % input_file)
			input_image = caffe.io.load_image(input_file)
			
			#Finding decaf features
			classifier.predict([input_image])
			blobs = classifier.blobs.items()
			features = blobs[-3][1].data[:,:,0,0]
			features_center = blobs[-3][1].data[4,:,0,0]
			features_center = np.resize(features_center, (1,4096))

			#Saving decaf features
			matfile = {}
			matfile['decaf'] = features
			matfile['decaf_center'] = features_center
			out_file = os.path.join(result_path, os.path.basename(input_file)+'.mat'), matfile
			sio.savemat(out_file)
			print matfile
			print "Decaf features calculated and saved at "+out_file

	except Exception as e:
		#In case of an error, print the whole error with traceback
		import traceback
		print str(traceback.format_exc()), socketid

