import torch
from tqdm import tqdm


def extract_features(model, dataloader, device, out_path='features.pt'):
    import itertools
    model.eval()
    features_map = {}
    with tqdm(desc='Extracting Features', unit='it', total=len(dataloader)) as pbar:
        for it, (ids, images) in enumerate(iter(dataloader)):
            images = images.to(device)
            features = model(images)
            for i, (id_i, f_i) in enumerate(zip(ids, features)):
                features_map[id_i] = f_i
            pbar.update()
    torch.save(features_map, open(out_path, 'wb'))
    return features_map
