#攻击方法实现程序
#包含FGSM BIM DEEPFOOL
import numpy as np
import torch
import torch.nn.functional as F

class Attacker:
    def __init__(self, clip_max=0.5, clip_min=-0.5):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass

class FGSM(Attacker):
    """
    Fast Gradient Sign Method
    Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015
    """
    #攻击初始化参数 eps：单次梯度下降步幅 clip_max整形图片最大值，clip_min整形图片最小值 超过边界则取边界值
    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        super(FGSM, self).__init__(clip_max, clip_min)
        self.eps = eps
    #定义类中的攻击生成函数，需求输入model：攻击的神经网络模型 x：原始干净图片 y:干净图片输出的结果
    def generate(self, model, x, y):
        #设定模型模式为测试（此模型下会恢复dropout中丢弃的链接）
        model.eval()
        #初始化对抗样本为干净图片并规整维度，之后才能直接与梯度相加减
        #nx = torch.unsqueeze(x, 0)
        #ny = torch.unsqueeze(y, 0)
        nx = x
        ny = y
        #声明NX需要梯度
        nx.requires_grad_()
        #使用模型得到当前对抗样本的输出
        out = model(nx)
        #计算损失函数，利用交叉熵
        loss = F.cross_entropy(out, ny)
        #梯度反向传播
        loss.backward()
        #根据得到的梯度按照FGSM的优化方法生成新的对抗样本
        x_adv = nx + self.eps * torch.sign(nx.grad.data)
        #规整图片确保数值有效
        x_adv.clamp_(self.clip_min, self.clip_max)
        #恢复之前的规整
        x_adv.squeeze_(0)
        #返回对抗样本 增加detach()函数确保该步骤不会对梯度产生影响
        return x_adv.detach()

class BIM(Attacker):
    """
    Basic Iterative Method
    Alexey Kurakin, Ian J. Goodfellow, Samy Bengio.
    Adversarial Examples in the Physical World.
    arXiv, 2016
    """
    def __init__(self, eps=0.15, eps_iter=0.01, n_iter=50, clip_max=0.5, clip_min=-0.5):
        super(BIM, self).__init__(clip_max, clip_min)
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter

    def generate(self, model, x, y):
        model.eval()
        #nx = torch.unsqueeze(x, 0)
        #ny = torch.unsqueeze(y, 0)
        nx=x
        ny=y
        nx.requires_grad_()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eta = torch.zeros(nx.shape).to(device)

        for i in range(self.n_iter):
            out = model(nx+eta)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps_iter * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()

class DeepFool(Attacker):
    """
    DeepFool
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
    DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks.
    CVPR, 2016
    """
    def __init__(self, max_iter=50, clip_max=0.5, clip_min=-0.5):
        super(DeepFool, self).__init__(clip_max, clip_min)
        self.max_iter = max_iter

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        #nx=x
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out = model(nx+eta)
        n_class = out.shape[1]
        out1=out.max(1)[1]
        out1=out1
        py = out1.item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out = model(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1
        
        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()
