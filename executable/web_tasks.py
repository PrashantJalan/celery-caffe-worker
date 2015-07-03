from __future__ import absolute_import
from executable.celery import app

@app.task
def add(x, y):
    return x + y

@app.task(ignore_result=True)
def mul(x,y):
    ans = x*y
    print ans
    return ans

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
def classifyImages(src_path, socketid, result_path, cpu=True):
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
	if cpu:
	    caffe.set_mode_cpu()
	    print "CPU mode"
	else:
	    caffe.set_mode_gpu()
	    print "GPU mode"

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

    	print("Classifying %d inputs." % len(inputs))

    	# Classify.
    	start = time.time()
    	prediction = classifier.predict(inputs)
	print len(prediction)
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
    	#in case of an error, print the whole error with traceback
	import traceback
        print str(traceback.format_exc()), socketid
