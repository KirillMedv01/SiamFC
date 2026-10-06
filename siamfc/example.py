from torch.utils.data import DataLoader
from datasets import Got10KValDataset

dataset = Got10KValDataset(
    root = '/Users/kirill/PycharmProjects/PythonProject2/siamfc/got10k_val'
)
dataloader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False)
print(len(dataloader), next(iter(dataloader)))