import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class CUNN(nn.Module):
    def __init__(self, input_size, shallow_feature_size, deep_feature_size):
        super(CUNN, self).__init__()
        self.shallow_feature_size = shallow_feature_size
        self.shallow_feature = nn.Sequential()
        self.shallow_feature.add_module('s_ln1', nn.Linear(input_size, 1024))
        self.shallow_feature.add_module('s_ln1_act', nn.ReLU(True))
        self.shallow_feature.add_module('s_ln2', nn.Linear(1024, 1024))
        self.shallow_feature.add_module('s_ln2_act', nn.ReLU(True))
        self.shallow_feature.add_module('s_ln3', nn.Linear(1024, 1024))
        self.shallow_feature.add_module('s_ln3_act', nn.ReLU(True))
        self.shallow_feature.add_module('s_ln4', nn.Linear(1024, self.shallow_feature_size))
        #self.shallow_feature.add_module('s_ln4_act', nn.ReLU(True))
        self.shallow_layer_norm = nn.LayerNorm(self.shallow_feature_size)

        self.deep_feature_size = deep_feature_size
        self.deep_feature = nn.Sequential()
        self.deep_feature.add_module('d_ln1', nn.Linear(self.shallow_feature_size, 256))
        self.deep_feature.add_module('d_ln1_act', nn.ReLU(True))
        self.deep_feature.add_module('d_ln2', nn.Linear(256, 128))
        self.deep_feature.add_module('d_ln2_act', nn.ReLU(True))
        self.deep_feature.add_module('d_ln3', nn.Linear(128, self.deep_feature_size))
        #self.deep_feature.add_module('d_ln3_act', nn.ReLU(True))
        self.deep_layer_norm = nn.LayerNorm(self.deep_feature_size)

        self.W = nn.Parameter(torch.rand(self.deep_feature_size, self.deep_feature_size))
        #self.W = nn.Parameter(torch.rand(self.shallow_feature_size, self.shallow_feature_size))

    def forward(self, observation, mode="optimization"):
        if mode == "optimization":
            s_feature = self.shallow_feature(observation)
            s_feature_norm = self.shallow_layer_norm(s_feature)
            d_feature = self.deep_feature(s_feature_norm)
            d_feature_norm = self.deep_layer_norm(d_feature)
            return d_feature_norm
        else:
            s_feature = self.shallow_feature(observation)
            s_feature_norm = self.shallow_layer_norm(s_feature)
            return s_feature_norm

    def sim(self, feature_1, feature_2):
        tmp = torch.matmul(self.W, feature_2.T)
        logits = torch.matmul(feature_1, tmp)
        #logits = torch.matmul(feature_1, feature_2.T)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class CU:
    def __init__(self, input_size, shallow_feature_size=256, deep_feature_size=64, temperature=0.5, device=None):
        self.device = device
        self.temperature = temperature

        self.cunn = CUNN(input_size, shallow_feature_size, deep_feature_size)
        self.cunn_target = CUNN(input_size, shallow_feature_size, deep_feature_size)
        self.cunn.to(self.device)
        #self.cunn.cuda()
        self.lr = 1e-4
        self.optimizer = optim.Adam(self.cunn.parameters(), lr=self.lr)
        self.loss = torch.nn.CrossEntropyLoss()

    def train(self, aug, ori_obss, aug_obss, num_epoch, args):
        ori_obss = torch.stack(ori_obss)
        batch_size = 32
        batch_num = args.num_processes * args.num_steps // batch_size
        ori_obss = torch.reshape(ori_obss, (batch_size, batch_num, -1))
        ori_obss = ori_obss.transpose(0, 1)

        aug_obss = torch.stack(aug_obss)
        aug_obss = torch.reshape(aug_obss, (batch_size, batch_num, -1))
        aug_obss = aug_obss.transpose(0, 1)

        for epoch in range(num_epoch):
            epoch_loss = 0
            for i in range(ori_obss.shape[0]):
                #aug_obs_1 = aug.gen_aug(ori_obs)
                #loss = self.update(ori_obs, aug_obs_1)

                ori_obs = ori_obss[i]
                aug_obs = aug_obss[i]
                #ori_obs = ori_obs.cuda()
                #aug_obs = aug_obs.cuda()
                loss = self.update(ori_obs, aug_obs)

                epoch_loss += loss
            if epoch % 9 == 0:
                print(epoch+1, " : ", epoch_loss)
            for params in self.optimizer.param_groups:
                params['lr'] *= 0.995

        for param, target_param in zip(self.cunn.parameters(), self.cunn_target.parameters()):
            target_param.data.copy_(0.1*param.data + 0.9*target_param.data)

    def update(self, aug_obs_1, aug_obs_2):
        deep_feature_1 = self.cunn(aug_obs_1, "optimization")
        #deep_feature_2 = self.cunn_target(aug_obs_2, "optimization")
        deep_feature_2 = self.cunn(aug_obs_2, "optimization")
        logits = self.cunn.sim(deep_feature_1, deep_feature_2)
        #print(logits)
        labels = torch.eye(deep_feature_1.shape[0], dtype=torch.float32).to(self.device)


        """deep_feature_1 = F.normalize(deep_feature_1, p=2, dim=1)
        deep_feature_2 = F.normalize(deep_feature_2, p=2, dim=1)

        labels = torch.eye(deep_feature_1.shape[0], dtype=torch.float32).to(self.device)
        logits = torch.div(torch.matmul(deep_feature_1, deep_feature_2.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits = torch.exp(logits)"""

        """logits_ab, logits_ba = self.compute_logits(deep_feature_1, deep_feature_2)
        #logits_ab = self.cunn.sim(deep_feature_1, deep_feature_2)
        #logits_ba = self.cunn.sim(deep_feature_2, deep_feature_1)
        #labels = torch.arange(logits_ab.shape[0]).long().to(self.device)
        #loss_ab = self.loss(logits_ab, labels)
        #loss_ba = self.loss(logits_ba, labels)"""

        loss = self.loss(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def encode_obs(self, ori_obs):
        with torch.no_grad():
            down_stream_feature = self.cunn(ori_obs, "encode")
        return down_stream_feature

    def compute_logits(self, feature_1, feature_2):
        feature_1 = feature_1 / torch.norm(feature_1)
        feature_2 = feature_2 / torch.norm(feature_2)
        logits_ab = torch.matmul(feature_1, feature_2.T)
        logits_ba = torch.matmul(feature_2, feature_1.T)
        #logits = logits - torch.max(logits, 1)[0][:, None]
        #print(logits)
        return logits_ab, logits_ba

    def to(self, device):
        self.cunn.to(device)

    def state_dict(self):
        return self.cunn.state_dict()

    def load_state_dict(self, path):
        self.cunn.load_state_dict(torch.load(path))
