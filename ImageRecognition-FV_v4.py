#!/usr/bin/env python

import pandas as pd
import numpy as np
import dicom
import scipy as sp
import os
import SimpleITK as sitk
import im_functions
import time

def remove_pixels(imgWhiteMatter9, line):
    img_final = imgWhiteMatter9
    y_size = img_final.GetSize()[1]
    x_size = img_final.GetSize()[0]
    lst=[]
    for j in range(0,y_size):
        lst_line = []
        for i in range(0,x_size):
            lst_line.append(img_final.GetPixel(i,j))
        lst.append(lst_line)
    unique, counts = np.unique(lst[line], return_counts=True)
    dic = {}
    for z in range(0,len(unique)):
        dic[unique[z]] = counts[z]
    lst_order = []
    for w in sorted(dic, key=dic.get, reverse=True):
        lst_order.append(w)   
    for j in range(0,y_size):
        for i in range(0,x_size):
            if img_final.GetPixel(i,j) != lst_order[1]:
                img_final.SetPixel(i,j,0)
    return img_final

def reject_outliers(data):
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return filtered

## Select just paths with images
path_home = '/disk/2/home/jupyter/mdamo/image/Bladder'
path_dir = os.path.join(path_home,'dataset') 
print path_dir, len(path_dir)
paths = [x[0] for x in os.walk(path_dir)]
imgOriginal = {}
#print paths
num_slashes = path_dir.count('/') + 3
num_count = [x.count("/") for x in paths]
lst_paths = []
lst_patient = []
start_digit = len(path_dir)+1
end_digit = start_digit+12
start_digit_fn = start_digit + 78


print 'Load data files'
for k in range(0,len(num_count)):
    if num_count[k] == num_slashes:
        temp_path = paths[k][start_digit:end_digit]
        lst_patient.append(temp_path)
        reader = sitk.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(paths[k])
        reader.SetFileNames(filenamesDICOM)
        idx_patient=paths[k][start_digit:end_digit]+'_'+paths[k][start_digit_fn:len(paths[k])]
        imgOriginal[idx_patient] = reader.Execute()
        print idx_patient

print imgOriginal.keys()

## Build the parameters for image treatment
## 0:Slice number, 1:lower threshold, 2:higher threshold,3:index to slice, 
# 4:size to slice, 5:list of seeds
# 224 for softmax
# 56 for cnn
size_width = 256
size_length = 256
position_x = 180
position_y = 180

parameters = {}
parameters['TCGA-4Z-AA80_1.3.6.1.4.1.14519.5.2.1.6354.4016.144142555551963557981353493486']=[142,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']
parameters['TCGA-4Z-AA80_1.3.6.1.4.1.14519.5.2.1.6354.4016.150758569648621263557606366803']=[71,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']
parameters['TCGA-4Z-AA7M_1.3.6.1.4.1.14519.5.2.1.6354.4016.162854436738149996894372386754']=[52,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a']
parameters['TCGA-4Z-AA7Y_1.3.6.1.4.1.14519.5.2.1.6354.4016.214661771464618386026097399472']=[144,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']
parameters['TCGA-4Z-AA7Y_1.3.6.1.4.1.14519.5.2.1.6354.4016.214847078197638251711809271842']=[46,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']
parameters['TCGA-ZF-AA5H_1.3.6.1.4.1.14519.5.2.1.1501.4016.164181128771533977140749314328']=[51,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b'] # Fake Class
parameters['TCGA-4Z-AA7S_1.3.6.1.4.1.14519.5.2.1.6354.4016.326998411825656269491445278688']=[122,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(60,60)],'T4a']
parameters['TCGA-4Z-AA81_1.3.6.1.4.1.14519.5.2.1.6354.4016.317640087404133002805900941943']=[79,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b'] 
parameters['TCGA-4Z-AA81_1.3.6.1.4.1.14519.5.2.1.6354.4016.320464949029552508421232616467']=[17,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b']
parameters['TCGA-4Z-AA86_1.3.6.1.4.1.14519.5.2.1.6354.4016.224907020407221461432739203142']=[98,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA86_1.3.6.1.4.1.14519.5.2.1.6354.4016.239007112060994142876156146112']=[19,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA82_1.3.6.1.4.1.14519.5.2.1.6354.4016.209104119021468002430837607983']=[32,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a'] 
parameters['TCGA-4Z-AA84_1.3.6.1.4.1.14519.5.2.1.6354.4016.691838776924230433830336107190']=[96,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA7W_1.3.6.1.4.1.14519.5.2.1.6354.4016.145274596694317152837372320897']=[65,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(60,60)],'T2a'] # -28,43
parameters['TCGA-4Z-AA7W_1.3.6.1.4.1.14519.5.2.1.6354.4016.993336843637173031286468487563']=[535,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']  

#Patient	Slice	Total	Contrast	mm	
#TCGA-4Z-AA7M	140	192	Y	 2.50 	T3a
#TCGA-4Z-AA7N	27	51	Y	 3.00 	T3a
#TCGA-4Z-AA7W	163	228	N	 2.50 	T2a
#TCGA-4Z-AA7W	582	708	N	 1.30 	T2a
#TCGA-4Z-AA70	101	117	Y	 2.50 	#N/A
#TCGA-4Z-AA7S	130	154	Y	 2.50 	T4a
#TCGA-4Z-AA7S	353	475	N	 2.50 	T4a
#TCGA-4Z-AA7Y	279	423	N	 1.30 	T2a
#TCGA-4Z-AA81	259	276	N	 2.50 	T2b
#TCGA-4Z-AA7Y	95	141	N	 3.80 	T2a
#TCGA-4Z-AA7Y	259	315	Y	 1.30 	T2a
#TCGA-4Z-AA80	326	468	N	 2.50 	T2a
#TCGA-4Z-AA80	163	234	N	 2.50 	T2a
#TCGA-4Z-AA81	297	376	N	 2.50 	T2b
#TCGA-4Z-AA82	268	300	Y	 2.50 	T2a
#TCGA-4Z-AA84	292	388	N	 2.50 	T3a
#TCGA-4Z-AA86	311	409	N	 2.50 	T3a
#TCGA-4Z-AA86	283	302	N	 2.50 	T3a

print 'Filter transformation CF and Threshold'

dic_imag = {}

for pat_num in parameters.keys():
    print pat_num
    obj_mri = imgOriginal[pat_num]
    for l in range(0,obj_mri.GetSize()[2]):
        # Smoothing
        imgOriginal_sl = obj_mri[:,:,l] # idxSlice
        imgSmooth = sitk.CurvatureFlow(image1=imgOriginal_sl,timeStep=0.125,numberOfIterations=5)
        v_lower= parameters[pat_num][1]
        v_upper= parameters[pat_num][2]
	imgWhiteMatter2=sitk.Shrink(imgSmooth,[2,2])
        imgWhiteMatter6 = sitk.Threshold(image1=imgWhiteMatter2,lower=v_lower,upper=v_upper,outsideValue=0)
        patient_num = pat_num + '_' + str(l)
        dic_imag[patient_num] = imgWhiteMatter6

del obj_mri
del imgOriginal
del reader

print 'Number of Images from all patients: ' + str(len(dic_imag.keys()))

## Build the vector to tensor flow  
#import itertools

x_lst_arr = []
x_arr = []
y_lst_arr = []
clas_name = []
clas_name_arr = []
cont = 0
acum = 0

print 'Cria os vetores para o TF'

for num_img in dic_imag.keys():
    img = dic_imag[num_img]
    if cont == 500:
        cont = 0
        acum += 500
        print(str(acum) + ': +500')
    x_arr = sitk.GetArrayFromImage(img).reshape((1,size_width*size_length))[0]
    x_lst_arr.append(x_arr)
    pixel_vector = []
    idx=num_img[0:77]
    if parameters[idx][6] == 'T2a':
        y_lst_arr.append([1,0,0,0])
	clas_name_arr.append('T2a')
    elif parameters[idx][6] == 'T2b':
        y_lst_arr.append([0,1,0,0])
	clas_name_arr.append('T2b')
    elif parameters[idx][6] == 'T3a':
        y_lst_arr.append([0,0,1,0])
	clas_name_arr.append('T3a')
    elif parameters[idx][6] == 'T4a':
        y_lst_arr.append([0,0,0,1])
	clas_name_arr.append('T4a')
    cont += 1

data = {'image':x_lst_arr,'label':y_lst_arr,'class':clas_name_arr}
df = pd.DataFrame(data=data)

size_x = len(x_lst_arr)
size_y = len(y_lst_arr)

size_train_x = int(round(size_x * 2/3))
size_test_x = size_x - size_train_x
size_train_y = int(round(size_y * 2/3))
size_test_y = size_y - size_train_y

batch_xs = np.array(x_lst_arr[0:size_train_x])
batch_ys = np.array(y_lst_arr[0:size_train_y])
batch_x_test = np.array(x_lst_arr[size_train_x:size_x])
batch_y_test = np.array(y_lst_arr[size_train_y:size_y])

print 'Roda o tensor flow para softmax'

import tensorflow as tf
#tf.reset_default_graph()

#Setup the model
with tf.device('/cpu:0'):
	dim = size_length * size_width
	cl = 4 #4 classes instead of 10
	var = 1
	x = tf.placeholder(tf.float32, [None, dim])
	W = tf.Variable(tf.zeros([dim, cl]))
	b = tf.Variable(tf.zeros([cl]))
	out_train = tf.Variable(tf.zeros([var]))
	out_test = tf.Variable(tf.zeros([var]))
        mm = tf.matmul(x,W)
        y = tf.nn.softmax(mm + b)
	y_ = tf.placeholder(tf.float32, [None, cl])
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Init Variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test})


# for 512 x 512 pixels we got 0.762 (?) of accuracy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

print 'Inicializa o CNN'

with tf.device('/gpu:0'):
        # First Convolution
	W_conv1 = weight_variable([7, 7, 1, size_width])
	b_conv1 = bias_variable([size_width])
	x_image = tf.reshape(x, [-1,size_width,size_length,1])

with tf.device('/cpu:0'):
	conv1 = conv2d(x_image, W_conv1)
	h_conv1 = tf.nn.relu(conv1 + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
        # Image with 256

with tf.device('/gpu:0'):
        # Second Convolution frame,frame,input,output
        W_conv2 = weight_variable([7, 7, size_width,256])
        b_conv2 = bias_variable([256])

with tf.device('/gpu:0'):
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
        #Image with 128

with tf.device('/gpu:0'):
        # Third Convolution
	W_conv3 = weight_variable([7, 7, 256,512])
        b_conv3 = bias_variable([512])

with tf.device('/gpu:0'):
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)
	#Image with 64
	
with tf.device('/gpu:0'):
	#Fourth Convolution
	W_conv4 = weight_variable([7, 7, 512,512])
        b_conv4 = bias_variable([512])

with tf.device('/gpu:0'):
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)
        #Image with 32
	h_pool5 = max_pool_2x2(h_pool4)
        #image with 16

with tf.device('/cpu:0'):
	# Densely Connected Layer
	W_fc1 = weight_variable([8*8*512,size_width*size_length])
	b_fc1 = bias_variable([size_width*size_length])
	h_pool1_flat = tf.reshape(h_pool5, [-1, 8*8*512])
	mat_mul_fc1 = tf.matmul(h_pool1_flat, W_fc1) + b_fc1

with tf.device('/gpu:0'):
	h_fc1 = tf.nn.relu(mat_mul_fc1)
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.device('/cpu:0'):
	# all the image for that W
	W_fc2 = weight_variable([size_width*size_length, cl])
	b_fc2 = bias_variable([cl])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy_value, accuracy_update_op = tf.contrib.metrics.streaming_accuracy(labels= tf.argmax(y_,1),predictions=tf.argmax(y_conv,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# collect batches of images before processing
BATCH_SIZE = 10
inter_train = int(np.ceil(len(batch_xs) / BATCH_SIZE))
inter_test = int(np.ceil(len(batch_x_test) / BATCH_SIZE))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333,allocator_type = 'BFC') 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
lst_acc_train = []
lst_acc_test = []
qtd_corrects_train = 0

with tf.device('/cpu:0'):
	init = tf.global_variables_initializer()
	sess.run(init)
	sess.run(tf.initialize_local_variables())
	loops = int(np.ceil(len(batch_xs) / BATCH_SIZE))
	start_time_total = time.time()
	for k in range(0,inter_train):
		print "Train Process: Loops n. times: "+str(k)+" each batch: "+str(BATCH_SIZE)
		ini = BATCH_SIZE*k
		if k != inter_train:
			end = BATCH_SIZE*(k+1)
		else:
			end = len(batch_xs)
		images = batch_xs[ini:end]
		labels = batch_ys[ini:end]
		start_time = time.time()
		train_accuracy = accuracy.eval(feed_dict={x:images, y_:labels, keep_prob: 1.0},session=sess)
		train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5},session=sess)
		sess.run([accuracy_update_op],feed_dict={x:images, y_:labels,keep_prob:0.5})
		duration = time.time() - start_time
                print "step %d, training accuracy %g, duration %d"%(k,train_accuracy,duration)
		print "Accuracy after batch %d: %f" % (k, accuracy_update_op.eval(feed_dict={x:images, y_:labels,keep_prob:1.0},session=sess))
		#print "AUC after batch %d: %f" % (k, auc_update_op.eval(feed_dict={x:images, y_:labels,keep_prob:1.},session=sess))
		lst_acc_train.append(train_accuracy)
	print "Accuracy after batchs %d: %f" % (k, accuracy_value.eval(session=sess))
	duration = time.time() - start_time_total
	print 'Duration: %f' % duration

np.save(path_dir + '/w_conv1.arr', sess.run(W_conv1))
np.save(path_dir + '/w_conv2.arr', sess.run(W_conv2))
np.save(path_dir + '/w_conv3.arr', sess.run(W_conv3))
np.save(path_dir + '/w_conv4.arr', sess.run(W_conv4))
np.save(path_dir + '/w_fc1.arr', sess.run(W_fc1))
np.save(path_dir + '/w_fc2.arr', sess.run(W_fc2))

sess.close()
