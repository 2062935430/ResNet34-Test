import os
from PIL import Image
from torchvision import transforms

# 创建文件夹
folder_path = './images'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 读取图像
image = Image.open('cat01.png')

# 定义数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，概率为50%
    transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5), # 随机旋转，角度范围为[-10, 10]，概率为50%
    transforms.RandomApply([transforms.RandomResizedCrop(256, scale=(0.8, 1.0))], p=0.5), # 随机缩放裁切，裁切后尺寸为256，缩放比例范围为[0.8, 1.0]，概率为50%
])

# 应用数据增强并保存图像
for i in range(4):
    transformed_image = transform(image)
    image_path = os.path.join(folder_path, f'transformed_image_{i}.jpg')
    transformed_image.save(image_path)