import torch
from tqdm import tqdm
from src.models.dl4mia_tissue_unet.dataset import TwoDimensionalDataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    data_dir = "data/processed/OCT_scans_new_20230419_512x512"
    data_type = "train"
    dataset = TwoDimensionalDataset(data_dir, data_type)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=True,
    )

    for i, sample in enumerate(tqdm(data_loader)):
        print(i)
