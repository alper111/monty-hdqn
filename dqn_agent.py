import os
import copy
import pickle

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyMemmapStorage, LazyTensorStorage


class DQNAgent:
    """Deep Q-learning agent."""

    def __init__(self,
                 network: torch.nn.Module,
                 device: str,
                 lr: float,
                 batch_size: int,
                 gamma: float = 0.99,
                 init_eps: float = 1.0,
                 min_eps: float = 0.1,
                 decay: float = 0.999,
                 decay_iter: int = 200,
                 target_update_iter: int = 100,
                 max_grad_norm: float = 1.0,
                 max_buffer_size: int = 1000000,
                 buffer_device: str = "disk",
                 save_folder: str = ".",
                 resume: bool = False):

        self.network = network.to(device)
        self.device = device
        self.batch_size = batch_size

        self.epsilon = init_eps
        self.min_eps = min_eps
        self.eps_decay_rate = decay
        self.eps_decay_iter = decay_iter
        self.target_update_iter = target_update_iter
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.iter = 0
        self.optimizer = torch.optim.Adam(lr=lr, params=self.network.parameters())
        self.criterion = torch.nn.MSELoss()
        self.target_net = copy.deepcopy(network)

        self.buffer_device = buffer_device
        self.max_buffer_size = max_buffer_size
        self.save_folder = save_folder

        if buffer_device != "disk":
            storage = LazyTensorStorage(max_buffer_size, device=buffer_device)
        else:
            scratch_dir = os.path.join(self.save_folder, "storage")
            storage = LazyMemmapStorage(max_buffer_size, scratch_dir=scratch_dir, existsok=True)

        self.buffer = ReplayBuffer(storage=storage,
                                   batch_size=batch_size,
                                   prefetch=8)
        if resume:
            self.load()

    def action(self, x, iv: tuple[int, ...] = None, epsilon: bool = True) -> int:
        x = to_tensor(x)
        with torch.no_grad():
            out = self.network(x.to(self.device))

        iv = to_tensor(iv)
        available_options, = torch.where(iv > 0.5)

        if epsilon and torch.rand(1) < self.epsilon:
            return int(torch.multinomial(iv, 1)[0])

        avail_max = out[available_options].argmax()
        return int(available_options[avail_max])

    def record(self, state, action, reward, next_state, terminated, init_vec, next_init_vec):
        sample = TensorDict(
            state=to_tensor(state),
            action=to_tensor(action, dtype=torch.int),
            reward=to_tensor(reward),
            next_state=to_tensor(next_state),
            terminated=to_tensor(terminated),
            init_vec=to_tensor(init_vec, dtype=torch.bool),
            next_init_vec=to_tensor(next_init_vec, dtype=torch.bool)
        )
        self.buffer.add(sample)

    def reset_buffer(self):
        self.buffer.empty()

    def loss(self, sample):
        sample = sample.to(self.device)
        batch_size = len(sample)

        with torch.no_grad():
            q_next = self.target_net(sample["next_state"])
            q_next[sample["next_init_vec"].logical_not()] = 0.0
            q_next_max, _ = q_next.max(dim=-1)
        q_next_max = q_next_max.reshape(-1)
        # NOTE: I'm currently just dividing the reward by a hundred
        # to regularize the network training. Anita also recommended
        # clamping it to 1.
        target = sample["reward"] / 100.0 + (1-sample["terminated"])*self.gamma*q_next_max
        q_now = self.network(sample["state"])[torch.arange(batch_size), sample["action"]]
        loss = self.criterion(q_now, target)
        return loss

    def update_target_net(self):
        self.target_net = copy.deepcopy(self.network)

    def update(self):
        if self.batch_size*10 <= len(self.buffer):
            sample = self.buffer.sample()
            loss = self.loss(sample)
            self.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"],
                                           max_norm=self.max_grad_norm)
            loss.backward()
            self.optimizer.step()
            self.iter += 1
            if self.iter % self.eps_decay_iter == 0:
                self.epsilon = max(self.epsilon * self.eps_decay_rate, self.min_eps)

            if self.iter % self.target_update_iter == 0:
                self.update_target_net()
            loss = loss.item()
        else:
            loss = 0.0
        return loss

    def save(self):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
            os.makedirs(os.path.join(self.save_folder, "storage"))
        name = os.path.join(self.save_folder, "network.pt")
        target_name = os.path.join(self.save_folder, "target_network.pt")
        optim_name = os.path.join(self.save_folder, "optim.pt")
        torch.save(self.network.eval().cpu().state_dict(), name)
        torch.save(self.target_net.eval().cpu().state_dict(), target_name)
        torch.save(self.optimizer.state_dict(), optim_name)
        self.network.to(self.device)
        self.target_net.to(self.device)
        self.buffer.dumps(self.save_folder)
        with open(os.path.join(self.save_folder, "info.pkl"), "wb") as f:
            pickle.dump({"epsilon": self.epsilon,
                         "iter": self.iter}, f)

    def load(self):
        name = os.path.join(self.save_folder, "network.pt")
        target_name = os.path.join(self.save_folder, "target_network.pt")
        optim_name = os.path.join(self.save_folder, "optim.pt")
        self.network.load_state_dict(torch.load(name))
        self.target_net.load_state_dict(torch.load(target_name))
        self.optimizer.load_state_dict(torch.load(optim_name))
        self.buffer.loads(self.save_folder)
        infodict = pickle.load(open(os.path.join(self.save_folder, "info.pkl"), "rb"))
        self.epsilon = infodict["epsilon"]
        self.iter = infodict["iter"]


def to_tensor(x, dtype=torch.float, device="cpu"):
    dv = torch.device(device)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype, device=dv)
    if x.dtype != dtype:
        x = x.type(dtype)
    if x.device != dv:
        x = x.to(dv)
    return x
