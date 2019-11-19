from torchvision.models import resnet101
import torch.nn as nn
class Age_Gender_Model(nn.Module):
  def __init__(self, pretrained = True, progress = True):
    super(Age_Gender_Model, self).__init__()
    self.model = resnet101(pretrained=pretrained, progress = progress)
    self.model.fc = nn.Identity()
    self.gender = nn.Linear(in_features=2048, out_features=2, bias=True)
    self.age = nn.Linear(in_features=2048, out_features=4, bias=True)

  def forward(self, x):
    x = self.model(x)

    gender = self.gender(x)
    age = self.age(x)
    return gender, age
