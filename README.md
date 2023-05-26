# 使用PyTorch实现图像分类
---
## 🥇定义模型：
  
### 🥈创建ResNet文件，建立一个小的神经网络
  
>以下为一个基于ResNet34模型的变种，其中定义了一个ResNet32类，  
>它包含了一个卷积层，四个残差层，一个平均池化层和一个全连接层，  
>每个残差层又包含了多个残差模块，每个残差模块由两个卷积层和一个快捷连接组成。  
  
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
            output = F.softmax(x,dim=1) # 设置dim = 1，对图像沿着某维度进行归一化，得到每张图片的概率分布或预测类别

            return output


    if __name__=='__main__':
        t = torch.randn([8, 3, 224, 224])
        model = ResNet32()
        out = model(t)
        print(out.shape)
        
        
在命令行调用该模型，显示结果为：
  
![ResNet结果](https://github.com/2062935430/ResNet34-Test/assets/128795948/bc8ac95a-58e7-40b8-a1d7-0bed798da981)  
  
由此可以看出该模型的输入张量是一个8x3x224x224的张量，  
表示有8个样本，每个样本有3个通道，每个通道有224x224个像素。  

该模型的输出张量是一个8x2的张量，  
表示有8个样本，每个样本有2个类别的概率，  
这是一个用于图像二分类任务的神经网络模型  
  
## 🥇图像分类模型的训练和测试
  
### 🥈训练过程(train)：
  
>🥉**验证集val与测试集test**  
>  
>相同点:  
>它们都不参与模型的训练过程，只用于评估模型的性能。  
>它们都需要和训练集有近似的数据分布，以保证模型的泛化能力。  
>  
>不同点:   
>验证集用于进一步确定模型中的超参数（例如正则项系数、ANN中隐含层的节点个数等），主要目的是为了挑选在验证集上表现最好的模型。  
>测试集只是用于评估模型的精确度（即泛化能力），主要目的是为了看看模型在实际生活中如何处理。  
>验证集是用来在训练过程中不断优化模型的，而测试集是用来在训练结束后最终评价模型的。  
>  
>目的：  
>验证集是用来在训练过程中评估模型的效果和调整模型的超参数的数据样本。  
>例如，可以使用验证集来选择最优的学习率、迭代次数、层数等。   
>   
>测试集是用来在训练结束后评估模型的性能和分类能力的数据样本。  
>例如，可以使用测试集来估计模型在真实场景中的泛化误差。  
>  
>验证集和测试集都不参与模型的拟合，但验证集会影响模型的选择，而测试集不会  
  
>🥉**超参数**  
>  
>超参数的优化是机器学习中一个重要的步骤，需要通过不同的策略来搜索最佳的配置，例如网格搜索、随机搜索、贝叶斯优化等。   
>   
>笼统而言，如果把训练过程比喻为做蛋糕，那我们会需要用到一些材料，比如面粉、鸡蛋、牛奶、糖等。  
>这些材料就相当于模型参数，它们是需要用数据来估计的，也就是需要根据不同的蛋糕食谱来确定其比例和数量。  
>    
>但我们还需要一些其他的东西，  
>比如烤箱的温度、烘焙的时间、蛋糕的大小和形状等等。 
>这些东西就相当于超参数，它们是你需要人为设定的，也就是需要根据自身经验与尝试来选择合适的值。  
>超参数的优化就是为了找到最适合做蛋糕的温度、时间、大小和形状等等。  
  
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
                
在以上代码import进了和train文件同一目录下的ResBet34模型，  
所以我们直接进入命令行调用train.py文件进行模型训练时，  
该训练会默认使用同一目录下的模型文件展开图像分类的训练过程，  
结果如下：  
  
![使用ResNet34模型完成train](https://github.com/2062935430/ResNet34-Test/assets/128795948/77384b99-e2a6-45fa-8cf4-4f88cdb99a13)
  
### 🥈测试过程（test）：  
