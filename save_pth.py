import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import glob
import os
import torchvision.transforms as tf



def read_pst_sample_data():
    file_path = '/data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/dataset/storage/000001.torch'  # Replace with your file's path
    content = torch.load(file_path)
    content = content[0]
    for key, value in content.items():
        if torch.is_tensor(value):
            print(f"Key: {key}, Tensor shape: {value.shape}")
        elif key == "images":
                print(f"length of images: {len(value)}")
        else:
            print(f"Key: {key}, Value: {value}")
    print(content['timestamps'])


dict_to_save = {}
dict_to_save['url'] = 'DAVIS'
dict_to_save['timestamps'] = []

cameras = []
images = []
masks = []

to_tensor = tf.ToTensor()


jpg_path = "/data/guest_storage/zhanpengluo/Dataset/DAVIS/rollerblade/JPEGImages/480p/rollerblade"
jpg_files =sorted(glob.glob(os.path.join(jpg_path, '*.jpg')))
npz_path = "/data/guest_storage/zhanpengluo/TrackAnyPoint/casualSAM/experiment_logs/01-09/davis_dev/rollerblade/BA_full"
npz_files = sorted(glob.glob(os.path.join(npz_path, '*.npz')))

mask_path = "/data/guest_storage/zhanpengluo/Dataset/DAVIS/rollerblade/toy_experiment/mask"
mask_files =sorted(glob.glob(os.path.join(mask_path, '*.png')))

assert len(jpg_files) == len(npz_files)

for i in range(len(jpg_files)):
    
    npz_file = npz_files[i]
    data = np.load(npz_file)
    K = data['K']
    cam = torch.zeros(18)
    cam[0] = float(K[0][0])
    cam[1] = float(K[1][1])
    cam[2] = float(K[0][2])
    cam[3] = float(K[1][2])
    cam[6] = 1.0
    cam[11] = 1.0
    cam[16] = 1.0
    cameras.append(cam)
    
    jpg_file = jpg_files[i]
    rgb = cv2.imread(jpg_file)[:, :, ::-1]
    rgb = cv2.resize(rgb,(640,360),interpolation=cv2.INTER_LINEAR)
    rgb = to_tensor(rgb)
    images.append(rgb)
    
    m   = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
    print(m.shape)
    masks.append(to_tensor(m))
    

    
    
cameras = torch.stack(cameras)

images = torch.stack(images)
masks = torch.stack(masks)

print(images.shape)
print(masks.shape)

dict_to_save['cameras'] = cameras
dict_to_save['images'] = images
dict_to_save['masks'] = masks
dict_to_save['key'] = 'rb4d_2'




torch.save(dict_to_save, '/data/guest_storage/zhanpengluo/FeedForwardGS/pixelsplat/dataset/davis4d/test/000001.torch')
