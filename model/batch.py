import torch


class Batch:
    def __init__(self, xs, ys):
        self.size = len(xs)
        self.xs = torch.FloatTensor(xs)
        self.ys = torch.LongTensor(ys)

    def to_device(self, device):
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)


def generate_batch(batch):
    """
    This function is use as an args 'collate_fn' of torch Data Loader
    Importance: do the padding here if needed
    """
    xs = []
    ys = []
    for (x, y) in batch:
        xs.append(x)
        ys.append(y)

    return Batch(xs, ys)
