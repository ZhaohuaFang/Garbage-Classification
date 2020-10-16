import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses
from tensorflow.keras.callbacks import EarlyStopping

from    load_figure import load_garbage, normalize, denormalize
from    resnet18 import ResNet18
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
sns.set()

tf.random.set_seed(22)
np.random.seed(22)
assert tf.__version__.startswith('2.')

def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    # 随机水平翻转
    x = tf.image.random_flip_left_right(x)
    x = tf.image.resize(x, [224, 224])
    # 再随机裁剪到合适尺寸
    x = tf.image.random_crop(x, [224,224,3])
    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    #y = tf.one_hot(y, depth=20)

    return x, y

# creat train db
images, labels, table = load_garbage('/content/drive/My Drive/垃圾3/garbage3',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(2000).map(preprocess).batch(200)
# crate validation db
images2, labels2, table = load_garbage('/content/drive/My Drive/垃圾3/garbage3',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(200)
# create test db
images3, labels3, table = load_garbage('/content/drive/My Drive/垃圾3/garbage3',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(200)

test_iter=iter(db_test)
sample=next(test_iter)
print(sample[0].shape)
print(sample[1].shape)
print(len(images2))
print(len(images3))

#在这里，使用了TensorFlow自带的网络结构和参数，当然也可以用自己训练的网络
#进行迁移学习
net = keras.applications.VGG16(weights='imagenet', include_top=False,
                               pooling='max')
net.trainable = False
newnet = keras.Sequential([
    net,
    layers.Dense(20)
])
newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=57)
newnet.evaluate(db_test)
newnet.save_weights('VGG16_garbage')
#newnet.load_weights('VGG16_garbage')
