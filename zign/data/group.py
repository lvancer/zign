from zign.data import zDataset

class zDatasetGroup(zDataset):

    def __init__(self, *datasets):
        super().__init__()
        
        self.datasets = datasets
        self.sizes = []
        for dataset in self.datasets:
            self.sizes.append(len(dataset))

        self.size = sum(self.sizes)

    def __getitem__(self, index):
        _index = index % self.size
        k = 0
        for i in range(len(self.sizes)):
            k = k + self.sizes[i]
            if k > _index:
                break
        j = _index - sum(self.sizes[0:i]) - 1
        dataset = self.datasets[i]
        return dataset[j]

    def __len__(self):
        return self.size