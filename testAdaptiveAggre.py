import torch
from model.adaptive_aggre import AdaptiveAggreModel
def main():
    #先定义两个张量
    tensor1 = torch.randn(2, 256)
    tensor2 = torch.randn(2, 256)
    #tensor3 = tensor1 * tensor2  # 元素直接相乘便可以得到哈达玛积

    #初始化自适应聚合模型
    model = AdaptiveAggreModel()
    parameter = model(tensor1,tensor2)
    print(parameter)

if __name__ == '__main__':
    main()