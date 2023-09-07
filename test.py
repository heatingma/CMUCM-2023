import torch
import torch.nn as nn

# 准备数据
input_data = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                           [0.2, 0.3, 0.4, 0.5],
                           [0.3, 0.4, 0.5, 0.6],
                           [0.4, 0.5, 0.6, 0.7]], dtype=torch.float32)

output_data = torch.tensor([[0.5],
                            [0.6],
                            [0.7],
                            [0.8]], dtype=torch.float32)

# 定义NARX模型
class NARXModel(nn.Module):
    def __init__(self):
        super(NARXModel, self).__init__()
        self.fc = nn.Linear(4, 10)
        self.lstm = nn.LSTM(10, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, input_data):
        x = torch.relu(self.fc(input_data))
        x = x.unsqueeze(1)
        import pdb
        pdb.set_trace()
        x, _ = self.lstm(x)
        pdb.set_trace()
        x = x.squeeze(1)
        x = self.output_layer(x)
        return x

# 创建NARX模型实例
model = NARXModel()

# 前向传播函数
def forward(input_data):
    return model(input_data)

# 打印预测结果
predictions = forward(input_data)
print(predictions)