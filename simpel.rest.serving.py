#-*- coding:utf-8 -*-

import tensorflow as tf 

from tensorflow import keras

import numpy as np 
import matplotlib.pyplot as  plt 

import os
import subprocess
import gzip
import random
import json
import requests

print(tf.__version__)
dirname,filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)




def load_data(path,kind='train'):
    

    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'%kind)

    with gzip.open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(),dtype= np.uint8,offset=8)

    with gzip.open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(),dtype= np.uint8,offset=16).reshape(len(labels),784)

    return images,labels



fashion_mnist = keras.datasets.fashion_mnist

data_path = './mnist-fashion/'
(train_images,train_labels) = load_data(data_path,'train')

(test_images,test_labels) = load_data(data_path,'t10k')

#scale to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

train_images = train_images.reshape(train_images.shape[0],28,28,1)
test_images = test_images.reshape(test_images.shape[0],28,28,1)
class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

print('\n train images shape {},of {}'.format(train_images.shape,train_images.dtype))
print('\n test images shape {},of {}'.format(test_images.shape,test_images.dtype))

def train2():
    model = keras.Sequential(
        [
        keras.layers.Conv2D(input_shape=(28,28,1),filters=8,kernel_size=3,strides=2,activation='relu',name='Conv1'),
        keras.layers.Flatten(),
        keras.layers.Dense(10,activation=tf.nn.softmax,name='Softmax')
        ]

    )


    testing =False
    epochs=1

    model.compile(optimizer=tf.train.AdamOptimizer(),loss= 'sparse_categorical_crossentropy',
    metrics=['accuracy'])

    model.fit(train_images,train_labels,epochs=epochs)


    test_loss,test_acc = model.evaluate(test_images,test_labels)

    print('\n Test accuracy: {}'.format(test_acc))



    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors,
    # and stored with the default serving key

    model_save_dir = './model/'
    if os.path.exists(model_save_dir):
        pass
    else:
        os.mkdir(model_save_dir)

    version =1

    export_path = os.path.join(model_save_dir,str(version))
    print('export_path= {}\n'.format(export_path))

    if os.path.isdir(export_path):
        print('\n model saved already,clean up\n')
        os.system('rm -r {export_path}')

    tf.saved_model.simple_save(session=keras.backend.get_session(),export_dir = export_path,
    inputs = {'input_images':model.input},
    #outputs = {t.name:t for t in model.outputs}
    outputs = {'output_labels':model.output}

    )
    #到底哪个，貌似结果一样的
    print(model.input,model.inputs,model.outputs,model.output)

    print('\n saved model')
    os.system('ls -l {export_path}')

    # WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/saved_model/simple_save.py:85: calling add_meta_graph_and_variables (from tensorflow.python.saved_model.builder_impl) with legacy_init_op is deprecated and will be removed in a future version.
    # Instructions for updating:
    # Pass your op to the equivalent parameter main_op instead.
    # INFO:tensorflow:Assets added to graph.
    # INFO:tensorflow:No assets to write.
    # INFO:tensorflow:SavedModel written to: /tmp/1/saved_model.pb

    ##当我们不知道这个模型的输入输出时，这个命令就很重要了 
    # the MetaGraphDefs (the models) and SignatureDefs (the methods you can call) in our SavedModel. 

    '''
    ~/anaconda3/envs/tf-gpu/bin/saved_model_cli 

    zkl@zkl-Z97X:~/zklcode/code/tfserve/test$ ~/anaconda3/envs/tf_gpu/bin/saved_model_cli show --dir ./model/1 --all

    MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

    signature_def['serving_default']:
    The given SavedModel SignatureDef contains the following input(s):
        inputs['input_images'] tensor_info:   #修改这里
            dtype: DT_FLOAT
            shape: (-1, 28, 28, 1)
            name: Conv1_input:0
    The given SavedModel SignatureDef contains the following output(s):
        outputs['Softmax/Softmax:0'] tensor_info:   outputs['output_labels'] tensor_info: 
            dtype: DT_FLOAT
            shape: (-1, 10)
            name: Softmax/Softmax:0  #但这里不变
    Method name is: tensorflow/serving/predict


    tensorflow_model_server -h
    '''

    #Serve your model with TensorFlow Serving
    #Add TensorFlow Serving distribution URI as a package source:
    # tf serving  用docker 是最方便的
    #用tf serving run，下载模型， 然后就可以做推断请求，uing REST 

'''
命令航执行

nohup tensorflow_model_server   --rest_api_port=8501   --model_name=fashion_model   
--model_base_path="/home/zkl/zklcode/code/tfserve/test/model/" >server.log 2>&1

'''


def requestclient():

    def show(idx, title):
        plt.figure()
        plt.imshow(test_images[idx].reshape(28,28))
        plt.axis('off')
        plt.title('\n\n{}'.format(title),fontdict={'size':16})



    rando = random.randint(0,len(test_images)-1)
    #show(rando,'exam:{}'.format(class_names[test_labels[rando]]))


    data = json.dumps( {"signature_name":"serving_default","instances":test_images[0:3].tolist() } )

    print('Data: {}...{}'.format(data[:50],data[len(data)-52:] ))

    headers = {"content-type":"application/json"}

    json_response = requests.post(url='http://localhost:8501/v1/models/fashion_model:predict',data=data,headers=headers)

    predictions = json.loads(json_response.text)['predictions']

    print(predictions)

    for i in range(0,3):
        show(0,'The model thought this was a {} (class {}),add it was actually a {} (class {})'
        .format(class_names[np.argmax(predictions[i])],test_labels[np.argmax(predictions[i])] ,
        class_names[i],test_labels[i] ))

    


if __name__ == '__main__':
    requestclient()