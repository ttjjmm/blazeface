from torch.utils.data import DataLoader
from data.widerface import WiderFaceDataset




def build_dataloader(config, mode='train'):


    cfg_data = config['dataset']
    cfg_loader = config['loader']
    cfg_data.update({'mode': mode})
    dataset = WiderFaceDataset(**cfg_data)


    dataloader = DataLoader(dataset, **cfg_loader, collate_fn=dataset.collate)

    return dataloader



