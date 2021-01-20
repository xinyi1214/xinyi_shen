from __future__ import print_function
import keras
from tensorflow.keras.models import Sequential #顺序模型
from tensorflow.keras.layers import Dense, Dropout, Flatten #引入各种层，dense可以用作线性回归，完成例如y=ax+b这样的模型
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt #image
import numpy as np
import glob as glob
from tqdm import tqdm
import cv2

batch_size = 60 #一次训练128个，因为太多电脑内存太大
#batch_size = 64
# nombre d'occurence(128) dans chaque batch
#num_classes = 3#所有的数字为10个，输出10个分类
epochs = 5 #训练12遍
#epochs = 1
# nombre de fois qu'il repasse dans le reseau



# input image dimensions
img_rows, img_cols = 150,90
#hauteur, largeur, et nombre de channels : canaux  | Channels : RedGreenBlue images NB : niveau de gris en 0 et 255
# X pour l'image
# Y pour l'appel
# image (h*L * C)  image(C*h*L)
# 3 personnes : un américain, un français, un allemand à un entretien d'embauche, à qui on demande de former une phrase avec pink green blue
# reconverti l'image en

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

dicoLocs ={'L001-huhu':0, 'L002-elgr':1, 'L004-sksk':2}
trainpath = glob.glob("/Users/shenxinyi/Documents/neurones/projet/try/*/train/*.wav")
testpath = glob.glob("/Users/shenxinyi/Documents/neurones/projet/try/*/test/*.wav")
x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

for path in trainpath:
  label = path.split('/')[1]
  x = cv2.imread(path, 0).astype(np.unit8)#imread: read the image, 0:black and white
  x = cv2.resize(x,(150, 90))
  y = dicoLocs.values() #change to number
  x_train_list.append(x)
  y_train_list.append(y)

for path in testpath:
  label = path.split('/')[1]
  x = cv2.imread(path, 0).astype(np.unit8)#imread: read the image, 0:black and white
  x = cv2.resize(x,(150, 90))
  y = dicoLocs.values() #change to number
  x_test_list.append(x)
  y_test_list.append(y)  

x_train = np.array(x_train_list)
#x_train = x_train.reshape(128, x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.astype('float32')
x_train/= 255

x_test = np.array(x_test_list)
#x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)
x_test = x_test.astype('float32')
x_test/= 255

labels = [l.split('/')[1] for l in glob.glob('images/*')]
num_classes = len(labels)

y_train = np.array(y_train_list)
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = np.array(y_test_list)
y_train = keras.utils.to_categorical(y_test, num_classes)




#train_image.shape 可以用来看多少张图片，以及图片大小(60000, 28, 28)

'''
if K.image_data_format() == 'channels_first': #channel是不是彩色，rgb为3，黑白为1， channelsfirst说明把channel放在高宽前，它是1，因为这是黑白图片
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# convert class vectors to binary class matrices 从0-9 变成 0-1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''

model = Sequential() #引入顺序模型
# 32 nb de neurone
# kernel : taille de filtre (3pixel, 3pixel)
model.add(Conv2D(128, kernel_size=(3, 3), #add是用来添加各种函数 
                 activation='relu',  #卷积层，训练32层卷积核， kernelsize，卷积核大小， 用relu做激活函数
                 input_shape=(150, 90, 1)))#第一层必须告诉它输入数据的形状
model.add(Conv2D(128, (3, 3), activation='relu')) # une autre couche，
model.add(MaxPooling2D(pool_size=(2, 2))) #un pooling : reduit la taill de l'image，用最大池化，大小2*2
model.add(Dropout(0.25)) # 防止过拟合，每次保留四分之一的隐藏单元
model.add(Flatten()) # passer à la classification de 1-9，把二维的数据一维化，因为原本是图片
model.add(Dense(128, activation='relu')) #输入层，每个映射是128个单元，用relu来激活
model.add(Dropout(0.5))#再随机扔掉一半的神经元
model.add(Dense(num_classes, activation='softmax'))#输出层，输出10个数字，用法softmax，多分类来激活，因为有多个分类
#model.summary() 可以显示模型的各种参数

#编译模型
model.compile(loss=keras.losses.categorical_crossentropy, #编译模型，loss是损失函数，用cate作为算法
              optimizer='sgd', #优化算法
              metrics=['accuracy']) # metrics可以看正确率    # Adadelta 比 sgd 好

#训练模型
model.fit(x_train, y_train,
          batch_size=batch_size, #一次训练多少张图片
          epochs=epochs, #训练多少次
          verbose=1, #为输出进度条记录
         # validation_data=(x_test, y_test)) #在训练过程中用测试数据进行测试，为了不出现过拟合的现象

#predictions = model.predict(x_test)#预测
#print(predictions[0])
# print('*****')
#print(np.argmax(predictions[11]))
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
