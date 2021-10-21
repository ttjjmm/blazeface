from torch.utils.data import Dataset, DataLoader
from data.widerface import WiderFaceDataset




def build_dataloader(config):


    cfg_data = config['dataset']
    cfg_loader = config['loader']

    dataset = WiderFaceDataset()


    dataloader = DataLoader(dataset, **cfg_loader, collate_fn=dataset.collate)


    pass




