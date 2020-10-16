import os, glob
import random, csv
import tensorflow as tf
import matplotlib.pyplot as plt
def load_csv(root, filename, name2label):
    # root:数据集根目录
    # filename:csv文件名
    # name2label:类别名编码表
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
        
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))

        #print(len(images), images)

        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            writer = csv.writer(f)
            for img in images:  
                name = img.split(os.sep)[-2]
                label = name2label[name]
               
                writer.writerow([img, label])
            #print('written into csv file:', filename)

    # read from csv file
    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            
            img, label = row
            label = int(label)

            images.append(img)
            labels.append(label)

    assert len(images) == len(labels)

    return images, labels


def load_garbage(root, mode='train'):
    name2label = {}  
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        # 给每个类别编码一个数字
        name2label[name] = len(name2label.keys())

    # 读取Label信息
    # [file1,file2,], [3,1]
    images, labels = load_csv(root, 'images.csv', name2label)

    if mode == 'train':  # 80%
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
#     elif mode == 'val':  # 10% = 80%->90%
#         images = images[int(0.8 * len(images)):int(0.9 * len(images))]
#         labels = labels[int(0.8 * len(labels)):int(0.9 * len(labels))]
    else:  # 10% = 90%->100%
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label


img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x, mean=img_mean, std=img_std):

    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):

    x = x * std + mean
    return x

def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [512, 512])
    
    # data augmentation, 0~255
    x= tf.image.random_flip_left_right(x)
    x = tf.image.random_crop(x, [512, 512, 3])
    
    
    x = tf.cast(x, dtype=tf.float32) / 255.    
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    return x, y
   
    
    
def main():
    import  time
    
    images, labels, table = load_garbage('D:\\Jupyter\\rubbish classification\\garbage', 'train')
    
    print(table)
    
    # images: string path
    # labels: number
    db = tf.data.Dataset.from_tensor_slices((images, labels))
    db = db.shuffle(2000).map(preprocess).batch(80)

    writter = tf.summary.create_file_writer('logs')
    try:
        for step, (x,y) in enumerate(db):
            with writter.as_default():


                #x = denormalize(x)
                tf.summary.image('img',x,step=step,max_outputs=9)
                time.sleep(5)
    except Exception:
        return None
if __name__ == '__main__':
    main()


# In[3]:


# #获取目录下文件的拓展名和文件名
# def file_name_listdir(file_dir):
#     for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
#         #分割文件名和扩展名
#         (file, ext) = os.path.splitext(files)
#         if(ext!='.jpg')&(ext!='.jpeg')&(ext!='.JPEG')&(ext!='.png')&(ext!='.JPG')&(ext!='.PNG'):
#             print(ext)
#             print(file)

# file_name_listdir("D:\Jupyter\垃圾分类\垃圾目录\有害垃圾")
