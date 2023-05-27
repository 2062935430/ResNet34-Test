# Implement image classification using PyTorch
---
## 🥇Define the model:
  
### 🥈Create a *ResNet* file and build a small neural network
  
The following is a variant based on the ResNet34 model, where a ResNet32 class is defined,  
which contains a convolutional layer, four residual layers, an average pooling layer and a fully connected layer,  
each residual layer contains multiple residual modules, each residual module consists of two convolutional layers and a shortcut connection.  
  
  
    import torch
    import torch.nn as nn
    import torch.nn.functional as 


     # 把残差连接补充到 Block 的 forward 函数中
     class Block(nn.Module):
        def __init__(self, dim, out_dim, stride) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_dim)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_dim)
            self.relu2 = nn.ReLU()

            def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            return x


    class ResNet32(nn.Module):
        def __init__(self, in_channel=64, num_classes=2):
            super().__init__()
            self.num_classes = num_classes
            self.in_channel = in_channel

            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
            self.maxpooling = nn.MaxPool2d(kernel_size=2)
            self.last_channel = in_channel

            self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
            self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
            self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
            self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)

            self.avgpooling = nn.AvgPool2d(kernel_size=2)
            self.classifier = nn.Linear(4608, self.num_classes)

        def _make_layer(self, in_channel, num_blocks, stride):
            layer_list = [Block(self.last_channel, in_channel, stride)]
            self.last_channel = in_channel
            for i in range(1, num_blocks):
                b = Block(in_channel, in_channel, stride=1)
                layer_list.append(b)
            return nn.Sequential(*layer_list)

        def forward(self, x):
            x = self.conv1(x)  # [bs, 64, 56, 56] 特征提取过程
            x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpooling(x)  
            x = x.view(x.shape[0], -1)  
            x = self.classifier(x)  
            output = F.softmax(x,dim=1) # 设置 dim = 1，沿某个维度对图像进行归一化，得到每个图像的概率分布或预测类别

            return output  


    if __name__=='__main__':
        t = torch.randn([8, 3, 224, 224])
        model = ResNet32()
        out = model(t)
        print(out.shape)
        
        
Call the model from the command line and display the results as:
  
![ResNet结果](https://github.com/2062935430/ResNet34-Test/assets/128795948/bc8ac95a-58e7-40b8-a1d7-0bed798da981)  
  
It can be seen that the input tensor of the model is an 8x3x224x224 tensor,  
which means there are 8 samples, each sample has 3 channels, each channel has 224x224 pixels.  
  
The output tensor of the model is an 8x2 tensor,   
which means there are 8 samples, each sample has 2 categories of probabilities,  
this is a neural network model for image binary classification task  
  
## 🥇Training and testing of image classification model
  
### 🥈Training：
  
>🥉**Validation set and test set**  
>  
>✒Similarities:  
>They both do not participate in the model training process,   
>and are only used to evaluate the model performance.  
>They both need to have a similar data distribution as the training set,  
>to ensure the model’s generalization ability.  
>  
>✒Differences:   
>The validation set is used to further determine the hyperparameters in the model，  
>(such as regularization coefficient, number of nodes in the hidden layer of ANN, etc.)  
>and the main purpose is to select the model that performs best on the validation set.  
>  
>The test set is only used to evaluate the accuracy (i.e., generalization ability) of the model,  
>and the main purpose is to see how the model handles real-life situations.  
>The validation set is used to optimize the model during the training process,  
>while the test set is used to evaluate the model after the training is finished.    
>  
>✒Purpose：  
>The validation set is a data sample used to evaluate the model effect,  
>and adjust the model hyperparameters during the training process.  
>  
>For example, you can use the validation set to select the optimal learning rate,  
>number of iterations, number of layers, etc.  
>  
>The test set is a data sample used to evaluate the model performance and classification ability after the training is finished.  
> For example, you can use the test set to estimate the model’s generalization error in real scenarios.  
>  
>***The validation set and test set do not participate in the model fitting,***  
>***but the validation set will affect the model selection, while the test set will not.***  
  
>🥉**Hyperparameters**  
>  
>Hyperparameter optimization is an important step in machine learning,  
>which requires different strategies to search for the best configuration,  
>such as grid search, random search, Bayesian optimization, etc.  
>   
>Generally speaking,  
>if we compare the training process to making a cake,  
>we will need some ingredients, such as flour, eggs, milk, sugar, etc.  
>  
>These ingredients are equivalent to model parameters,  
>which need to be estimated with data,  
>that is, they need to be determined according to different cake recipes.  
>  
>But we also need some other things,  
>such as the oven temperature, baking time, cake size and shape, etc.  
>These things are equivalent to hyperparameters, which you need to set manually,  
>that is, you need to choose appropriate values based on your own experience and trial and error.  
>  
>Hyperparameter optimization is to find the most suitable temperature, time, size and shape for making a cake.
  
    import argparse
    import time
    import json
    import os
    import ResNet34

    from tqdm import tqdm
    from models import *
    # from efficientnet_pytorch import EfficientNet
    from torch import nn
    from torch import optim
    # from torch.optim.lr_scheduler import *
    from torchvision import transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from tools import warmup_lr


    # 初始化参数
    def get_args():
        """在下面初始化你的参数.
        """
        parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

        # exp
        parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
        parser.add_argument('--train_dir', type=str, default='data/train')
        parser.add_argument('--val_dir', type=str, default='data/val')
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--save_station', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--is_mps', type=bool, default=False)
        parser.add_argument('--is_cuda', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--test_batch_size', type=int, default=4)
        parser.add_argument('--lr', type=float, default=0.001)

        # dataset
        parser.add_argument('--num_classes', type=int, default=2)
        parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
        parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

        # model
        parser.add_argument('--model', type=str, default='ResNet18',
                            choices=[
                                'ResNet18',
                                'ResNet34',
                                'ResNet50',
                                'ResNet18RandomEncoder',
                            ])

        # scheduler
        parser.add_argument('--warmup_epoch', type=int, default=1)

        # 通过json记录参数配置
        args = parser.parse_args()
        args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
        log_file = os.path.join(args.directory, 'log.json')
        if not os.path.exists(args.directory):
            os.makedirs(args.directory)
        with open(log_file, 'w') as log:
            json.dump(vars(args), log, indent=4)

        # 返回参数集
        return args


    class Worker:
        def __init__(self, args):
            self.opt = args

            # 判定设备
            self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
            kwargs = {
                'num_workers': args.num_workers,
                'pin_memory': True,
            } if args.is_cuda else {}

            # 载入数据
            train_dataset = datasets.ImageFolder(
                args.train_dir,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(256),
                    transforms.ToTensor()
                    # transforms.Normalize(opt.data_mean, opt.data_std)
                ])
            )
            val_dataset = datasets.ImageFolder(
                args.val_dir,
                transform=transforms.Compose([
                    transforms.RandomResizedCrop(256),
                    transforms.ToTensor()
                    # transforms.Normalize(opt.data_mean, opt.data_std)
                ])
            )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=args.test_batch_size,
                shuffle=False,
                **kwargs
            )

            # 挑选神经网络、参数初始化
            net = None
            if args.model == 'ResNet18':
                net = ResNet18(num_cls=args.num_classes)
            elif args.model == 'ResNet34':
                net = ResNet34(num_cls=args.num_classes)
            elif args.model == 'ResNet50':
                net = ResNet50(num_cls=args.num_classes)
            elif args.model == 'ResNet18RandomEncoder':
                net = ResNet18RandomEncoder(num_cls=args.num_classes)
            assert net is not None

            self.model = net.to(self.device)

            # 优化器
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=args.lr
            )

            # 损失函数
            self.loss_function = nn.CrossEntropyLoss()

            # warm up 学习率调整部分
            self.per_epoch_size = len(train_dataset) // args.batch_size
            self.warmup_step = args.warmup_epoch * self.per_epoch_size
            self.max_iter = args.epochs * self.per_epoch_size
            self.global_step = 0

    def train(self, epoch):
            self.model.train()
            bar = tqdm(enumerate(self.train_loader))
            for batch_idx, (data, target) in bar:
                self.global_step += 1
                data, target = data.to(self.device), target.to(self.device)

                # 训练中...
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                lr = warmup_lr.adjust_learning_rate_cosine(
                    self.optimizer, global_step=self.global_step,
                    learning_rate_base=self.opt.lr,
                    total_steps=self.max_iter,
                    warmup_steps=self.warmup_step
                )

                # 更新进度条
                bar.set_description(
                    'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item(),
                        lr
                    )
                )
            bar.close()

        def val(self):
            self.model.eval()
            validating_loss = 0
            num_correct = 0
            with torch.no_grad():
                bar = tqdm(self.val_loader)
                for data, target in bar:
                    # 测试中...
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                    pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                    num_correct += pred.eq(target.view_as(pred)).sum().item()
                bar.close()

            # 打印验证结果
            validating_loss /= len(self.val_loader)
            print('val >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
                validating_loss,
            num_correct,
            len(self.val_loader.dataset),
            100. * num_correct / len(self.val_loader.dataset))
            )

            # 返回重要信息，用于生成模型保存命名
            return 100. * num_correct / len(self.val_loader.dataset), validating_loss


    if __name__ == '__main__':
        # 初始化
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(0)
        args = get_args()
        worker = Worker(args=args)

        # 训练与验证
        for epoch in range(1, args.epochs + 1):
            worker.train(epoch)
            val_acc, val_loss = worker.val()
            if epoch > args.save_station:
                save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                           % (args.model, epoch, val_acc, val_loss)
                torch.save(worker.model, save_dir)
                
In the above code, a ResBet34 module that is stored in the same directory as train.py is imported,  
so when you directly enter the command line program to call the train.py file to train the model,  
the training will use the same directory of the model to carry out image classification training by default,   
the output of the training process is as follows:  
  
![使用ResNet34模型完成train](https://github.com/2062935430/ResNet34-Test/assets/128795948/77384b99-e2a6-45fa-8cf4-4f88cdb99a13)
  
---
### 🥈Testing： 
  
During the training process, we completed the training of the model,  
and assigned the save-station of the train script to 1,  
then our model will start saving from the first round, and then save the model training parameters of 5 rounds in the default path,  
in this code, a dictionary folder will be created in the current directory for the ResNet model saving work.  
  
In the testing process, we need to test these saved models,  
refer to the val function of train.py to design the test script,  
evaluate the model performance on the validation set, including accuracy, loss and other indicators.  
  
    import torch
    import ResNet34
  
    from torchvision import transforms
    from torchvision import datasets
    from torch.utils.data import DataLoader
  
    # 指定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
    # 加载训练好的模型
    model_path = "dictionary/ResNet34/Hi05-24-20-26/ResNet34-epochs-5-model-val-acc-98.000-loss-0.000000..pt"
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    # 指定数据集
    test_dir = "data/test"  # 你的测试数据集的路径
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.ToTensor()
        ])
    )

    # 指定数据集加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,  # 你的测试批次大小
        shuffle=False,
        num_workers=4  # 你的数据加载线程数
    )
  
    # 定义损失函数
    loss_function = torch.nn.CrossEntropyLoss()
  
  
    # 定义一个完整的val模型，参考train.py中的val函数
    def val():
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # 测试中...
                data, target = data.to(device), target.to(device)
                output = model(data)
                validating_loss += loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
  
                # 打印验证结果
                validating_loss /= len(test_loader)
                print('test >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
                    validating_loss,
                    num_correct,
                    len(test_loader.dataset),
                    100. * num_correct / len(test_loader.dataset))
                )

  
    # 调用val模型进行测试
    if __name__ == '__main__':
        val()
  
  
This test code block loads the trained ResNet34 model,  
and puts it on the specified device,  
sets it to evaluation mode, and does not update the gradients.  
  
The path of the test dataset is specified as “data/test”,  
and the ImageFolder class is used to load the image data, at the same time,  
using RandomResizedCrop and ToTensor these two transformations to crop and convert the images to tensors.  
The dataset loader is specified as test_loader, setting the batch size to 4, not shuffling the data order, using 4 threads to load the data.
  
He defined a val function, which is used to evaluate the model performance on the test dataset,  
calculate the average loss and accuracy, and print the results, finally calling the val function in the main function to test.  
  
![测试结果显示出两轮测试的平均损失与批次准确率](https://github.com/2062935430/ResNet34-Test/assets/128795948/c96e3d26-7b3f-4651-a538-5c44ceaba04a)
  
As shown in the figure, you can learn about the model performance on the test set through test.  
The model accuracy on the first batch is 66.667%, and on the second batch is 100.000%.  
