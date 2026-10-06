from model import SiamFCHead, Net, Krizhevsky_net
from torch.utils.data import DataLoader
from datasets import Got10KValDataset, create_labels
import torch.optim as optim
from torch import nn as nn

backbone = Krizhevsky_net()
head = SiamFCHead(out_scale=1.0)
net = Net(backbone, head)

optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

dataset = Got10KValDataset(
    root = '/Users/kirill/PycharmProjects/PythonProject2/siamfc/got10k_val'
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for epoch in range(10):
    net.train()
    for d in dataloader:
        z, x = d['template'], d['search']
        offsets = d['offset']
        out = net(z, x)
        out = out.squeeze(1)

        labels, weights = create_labels(out.shape[-2:],
                                        offsets)
        loss = (criterion(out, labels) * weights).sum() / out.size(0)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch} loss {loss.item()}')
