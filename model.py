from abc import abstractmethod, ABC
from typing import Any
from torch import device, nn
import torch
import torch.nn.functional as f
from typing import TypeVar


class BaseEncoder(ABC, nn.Module):
    def __init__(self, embedding_dim=256):
        super(BaseEncoder, self).__init__()

        self.emb_dim = embedding_dim
        self.encoder = None

    def get_embegging_dim(self):
        return self.emb_dim
    
    @abstractmethod
    def forward(self, x) -> Any:
        raise NotImplemented("Base encoder class' forward method is not implemented")


class FCEncoder(BaseEncoder):
    def __init__(self, state_dim: tuple[int], embedding_dim=256):
        super(FCEncoder, self).__init__(embedding_dim)
        self.input_fc = nn.Linear(state_dim[0], embedding_dim)

    def forward(self, x):
        x = f.relu(self.input_fc(x))
        return x


class CNNEncoder(BaseEncoder):
    def __init__(self, image_shape: tuple[int, int, int], embedding_dim:int = 128):
        def conv2d_out_shape(conv: nn.Conv2d, size: tuple[int, int]):
            kernel_size = conv.kernel_size
            stride = conv.stride
            
            new_width = (size[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1
            new_height = (size[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1
            return new_width, new_height

        super(CNNEncoder, self).__init__(embedding_dim)
        
        image_height, image_width, in_channels = image_shape
        # assert image_width == image_height, "CNN only works with square images"
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=6, stride=1)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv1_size = conv2d_out_shape(self.conv1, (image_width, image_height))
        self.conv3_size = conv2d_out_shape(self.conv3, self.conv1_size)
        self.linear_output_size = self.conv3_size[0] * self.conv3_size[1] * self.bn3.num_features

        self.output_fc = nn.Linear(self.linear_output_size, embedding_dim)

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = f.relu(self.conv1(x))
        x = f.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = f.relu(self.output_fc(x))
        return x


class DeepQNet(nn.Module):
    def __init__(self, encoder: BaseEncoder, actions_dim:int):
        super(DeepQNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = encoder
        self.output_fc = nn.Linear(self.encoder.get_embegging_dim(), actions_dim)

        self.to(self.device)

    def get_device(self):
        return self.device

    def forward(self, x):
        x = self.encoder(x)
        x = self.output_fc(x)
        return x


BaseEncoderType = TypeVar("BaseEncoderType", bound=BaseEncoder)


# Simple test to check that all is working
if __name__ == "__main__":
    fc_encoder = FCEncoder(12)
    fc_qnet = DeepQNet(fc_encoder, 2)
    print(fc_qnet(torch.randn(12).unsqueeze(0)))
    
    cnn_encoder = CNNEncoder((3, 128, 128))
    cnn_qnet = DeepQNet(cnn_encoder, 2)
    print(cnn_qnet(torch.randn((3, 128, 128)).unsqueeze(0)))

    
