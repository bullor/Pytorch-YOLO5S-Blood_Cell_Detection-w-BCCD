# [Pytorch-YOLO5S]Blood_Cell_Detection w/ BCCD

Here you can find my work to train and detect cells with Yolov5 with BCCD_Dataset.

In this notebook,I implemented a YOLO on BCCD dataset consisting of cell images and I trained the YOLO for object detection / classification purposes.

Below steps were applied through model build-up :

- Select smaller scale Pytorch YOLOV5 algorithm to implement training and detection to use computational resources efficiently.
![YOLOv5s](https://user-images.githubusercontent.com/107344116/188861705-c8b880d7-417d-4b10-baad-8bdba2c98b58.png)
- Download the BCCD Data Set.
- Parse XML annotations to Pandas Dataframe to manipulate the data.
- To train the Yolov5 model, .txt files are produced with bounding box features and encoded label which is compatible with Yolov5 by using preprocessing techniques including normalization and Sklearn LabelEncoder tool.
- Train the model and observe the results in Tensorboard.
- Model Performance Scores: Precision: 0.0899 | Recall 0.827 | 
mean average precision 0.0879 |
- Output txt labels are reflected to images to verify whether model output can be visualized.

# Data Preprocessing and Manipulation

```python
import os, sys, random, shutil
import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import patches
import numpy as np

annotations = sorted(glob('/content/drive/My Drive/BCCD_Dataset/BCCD/Annotations/*.xml'))

df = []
cnt = 0
for file in annotations:
  prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
  filename = str(cnt) + '.jpg'
  row = []
  parsedXML = ET.parse(file)
  for node in parsedXML.getroot().iter('object'):
    blood_cells = node.find('name').text
    xmin = int(node.find('bndbox/xmin').text)
    xmax = int(node.find('bndbox/xmax').text)
    ymin = int(node.find('bndbox/ymin').text)
    ymax = int(node.find('bndbox/ymax').text)

    row = [prev_filename, filename, blood_cells, xmin, xmax, ymin, ymax]
    df.append(row)
  cnt += 1

data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

data[['prev_filename','filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('/content/drive/My Drive/blood_cell_detection.csv', index=False)
data.head(10)
```
![datahead](https://user-images.githubusercontent.com/107344116/188862702-11ccdf39-d392-4d1d-879c-c4450728c1ad.png)

Normalization is applied to annotations. Below code will convert the annotations in (1,1) plane.

```python
img_width = 640
img_height = 480
def width(df):
  return int(df.xmax - df.xmin)
def height(df):
  return int(df.ymax - df.ymin)
def x_center(df):
  return int(df.xmin + (df.width/2))
def y_center(df):
  return int(df.ymin + (df.height/2))
def w_norm(df):
  return df/img_width
def h_norm(df):
  return df/img_height

df = pd.read_csv('/content/drive/My Drive/blood_cell_detection.csv')

le = preprocessing.LabelEncoder()
le.fit(df['cell_type'])
print(le.classes_)
labels = le.transform(df['cell_type'])
df['labels'] = labels

df['width'] = df.apply(width, axis=1)
df['height'] = df.apply(height, axis=1)

df['x_center'] = df.apply(x_center, axis=1)
df['y_center'] = df.apply(y_center, axis=1)

df['x_center_norm'] = df['x_center'].apply(w_norm)
df['width_norm'] = df['width'].apply(w_norm)

df['y_center_norm'] = df['y_center'].apply(h_norm)
df['height_norm'] = df['height'].apply(h_norm)

df.head(30)
```
![data30](https://user-images.githubusercontent.com/107344116/188863455-acc4f85b-712b-4c85-892a-c137a05718ac.png)

All dataframe rows are iterated and saved to .txt files as labels of bounding boxes.

```python
df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
print(df_train.shape, df_valid.shape)

os.mkdir('/content/drive/My Drive/bcc/')
os.mkdir('/content/drive/My Drive/bcc/images/')
os.mkdir('/content/drive/My Drive/bcc/images/train/')
os.mkdir('/content/drive/My Drive/bcc/images/valid/')

os.mkdir('/content/drive/My Drive/bcc/labels/')
os.mkdir('/content/drive/My Drive/bcc/labels/train/')
os.mkdir('/content/drive/My Drive/bcc/labels/valid/')

def segregate_data(df, img_path, label_path, train_img_path, train_label_path):
  filenames = []
  for filename in df.filename:
    filenames.append(filename)
  filenames = set(filenames)
  
  for filename in filenames:
    yolo_list = []

    for _,row in df[df.filename == filename].iterrows():
      yolo_list.append([row.labels, row.x_center_norm, row.y_center_norm, row.width_norm, row.height_norm])

    yolo_list = np.array(yolo_list)
    txt_filename = os.path.join(train_label_path,str(row.prev_filename.split('.')[0])+".txt")
    # Save the .img & .txt files to the corresponding train and validation folders
    np.savetxt(txt_filename, yolo_list, fmt=["%d", "%f", "%f", "%f", "%f"])
    shutil.copyfile(os.path.join(img_path,row.prev_filename), os.path.join(train_img_path,row.prev_filename))
 
## Apply function ## 
src_img_path = "/content/drive/My Drive/BCCD_Dataset/BCCD/JPEGImages/"
src_label_path = "/content/drive/My Drive/BCCD_Dataset/BCCD/Annotations/"

train_img_path = "/content/drive/My Drive/bcc/images/train"
train_label_path = "/content/drive/My Drive/bcc/labels/train"

valid_img_path = "/content/drive/My Drive/bcc/images/valid"
valid_label_path = "/content/drive/My Drive/bcc/labels/valid"

segregate_data(df_train, src_img_path, src_label_path, train_img_path, train_label_path)
segregate_data(df_valid, src_img_path, src_label_path, valid_img_path, valid_label_path)

print("No. of Training images", len(os.listdir('/content/drive/My Drive/bcc/images/train')))
print("No. of Training labels", len(os.listdir('/content/drive/My Drive/bcc/labels/train')))

print("No. of valid images", len(os.listdir('/content/drive/My Drive/bcc/images/valid')))
print("No. of valid labels", len(os.listdir('/content/drive/My Drive/bcc/labels/valid')))
```
![samples](https://user-images.githubusercontent.com/107344116/188863891-f16f6bb8-f686-413f-8b78-743f780fc2b0.png)

```python
!git clone 'https://github.com/ultralytics/yolov5.git'
```
![clone](https://user-images.githubusercontent.com/107344116/188864285-771b7923-0114-4557-b42f-2f691d56f628.png)

bcc.yaml file is constructed and saved into the directory then this file is copied as bcc.yaml file inside /yolov5 directory. Inside bcc.yaml the configuration is as follows for class numbers an items. After training starts then cfg file will be overwrited with that yaml file.

![bcc](https://user-images.githubusercontent.com/107344116/188864473-086d34fa-5341-4f09-9b47-56bec0e22bf3.png)

Dependencies for algorithm are installed.

```python
!pip install -qr '/content/drive/My Drive/yolov5/requirements.txt'  # install dependencies
shutil.copyfile("bcc.yaml", '/content/drive/My Drive/yolov5/bcc.yaml')
```
# Training :
We need to configure the training parameters such as no.of epochs, batch_size, etc.,

Training Parameters
!python
- <'location of train.py file'>
- --img <'width of image'>
- --batch <'batch size'>
- --epochs <'no of epochs'>
- --data <'location of the .yaml file'>
- --cfg <'Which yolo configuration you want'>(yolov5s/yolov5m/yolov5l/yolov5x).yaml | (small, medium, large, xlarge)
- --name <'Name of the best model to save after training'>

```python
!python yolov5/train.py --img 640 --batch 8 --epochs 100 --data bcc.yaml --cfg yolov5/models/yolov5s.yaml --name BCCM
```

![trainingresult](https://user-images.githubusercontent.com/107344116/188865155-9bb03c46-0bb6-45f1-b30b-a1ef2a2ef080.png)

METRICS FROM TRAINING PROCESS
No of classes, No.of images, No.of targets, Precision (P), Recall (R), mean Average Precision (map)
- Class | Images | Targets |    P   |   R   | mAP@.5 | mAP@.5:.95: |
- all   | 270    |     489 | 0.0899 | 0.827 | 0.0879 |  0.0551

So from the values of P (Precision), R (Recall), and mAP (mean Average Precision) we result whether our model is doing well for our dataset. Although, I trained the model for only 100 epochs, the performance is great.

Also, we can view the logs in tensorboard after training is completed by activating tensorboard with below commands. logdir is yolov5/runs/train/ in my case for my model.

```python
%load_ext tensorboard
%tensorboard --logdir yolov5/runs/train/
```

![tensorboard](https://user-images.githubusercontent.com/107344116/188866310-2bb18c3e-a817-4f84-85c9-d7e8b4cc9ee4.png)

With below code prediction is done to all validation images and results are saved to project destination. Best weight is used for validation after training phase.

Inference parameters:

!python
- <'location of detect.py file'>
- --source <'location of image/ folder to predict'>
- --weight <'location of the saved best weights'>
- --project <'location to store the outputs after prediction'>
- --img-size <'Image size of the trained model'>

## And there are more customization params available, please kindly check them in the detect.py file. ##

```python
!python yolov5/detect.py --source './yolov5/datasets/bcc/images/valid/' --weights './yolov5/runs/train/BCCM/weights/best.pt' --project './yolov5/inference/output/'
```

![detector](https://user-images.githubusercontent.com/107344116/188866770-77d61009-a4d7-4497-b02a-7a4c8e1be066.png)

After I get inference results to yolov5/inference/output/exp path, in below code snippet, I showed 9 inference result samples. Results are shown with class labels, confidence score and bounding boxes.

```python
import os
import numpy as np
from PIL import ImageFile, Image
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

folder_dir = './yolov5/inference/output/exp'
files = os.listdir(folder_dir)
i=0
fig, axes = plt.subplots(3, 3,figsize=(20,20))
for i, img_name in enumerate(files[:]):
  im = mpimg.imread(folder_dir + '/' + img_name)
  axes[i//3, i%3].imshow(im)
  if i==8:
    break
plt.show()
```

![output1](https://user-images.githubusercontent.com/107344116/188867553-4a195557-2de8-4300-8cd0-46ea4b24a6c6.png)


![2](https://user-images.githubusercontent.com/107344116/188867576-38136c54-8cff-4980-8c0c-e5ff89345d5a.png)


![output3](https://user-images.githubusercontent.com/107344116/188867524-f0fb06a3-5814-4055-baaf-0584574df0ae.png)

# Convey outputs from the .txt file to image :

Also, we can save the output to a .txt file, which contains some of the input image’s bbox co-ordinates. Run the below code, to get the outputs in .txt file,

```python
!python yolov5/detect.py --source './yolov5/datasets/bcc/images/valid/' --weights './yolov5/runs/train/BCCM/weights/best.pt' --project './yolov5/inference/outputtxt/ --view-img --save-txt
```

After running the code, we can see the outputs in the below path. File has class x_center_norm y_center_norm width_norm height_norm

![label1](https://user-images.githubusercontent.com/107344116/188868044-9dbe0555-0520-410a-9c78-f9bc8ad85e9d.png)

# Code Snippet For Running Model As Production Model
Below code snippet which shows one predicted pictures with all system path and library requirements. My model and programs run accordingly.After running for image, it shows detection result class, bounding box and confidence score. Now model can be given to production. Any questions are welcome regarding training and inference. Thank you.

```python
import os, sys, random
from glob import glob
import matplotlib.pyplot as plt
%matplotlib inline
!pip install -qr '/content/drive/My Drive/yolov5/requirements.txt'  # install dependencies

## Add the path where you have stored the neccessary supporting files to run detect.py ##
## Replace this with your path.##
sys.path.insert(0, '/content/drive/My Drive/yolov5') 
print(sys.path)
cwd = os.getcwd()
print(cwd)

## Single Image prediction
## Beware the contents in the output folder will be deleted for every prediction
#source './yolov5/datasets/bcc/images/valid/'
!python yolov5/detect.py --source './BCCD_Dataset/BCCD/JPEGImages/BloodImage_00000.jpg' --weights './yolov5/runs/train/BCCM/weights/best.pt' --project './yolov5/inference/output/production' --device 'cpu'
img = plt.imread('./yolov5/inference/output/production/exp3/BloodImage_00000.jpg')
plt.imshow(img)
```

![output11](https://user-images.githubusercontent.com/107344116/188868827-3f8b419f-1dc3-44ab-986b-1c8ada684e37.png)

