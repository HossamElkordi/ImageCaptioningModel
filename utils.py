import json
from tqdm import tqdm


def extract_features(model, dataloader, device, out_path='features.json'):
    model.eval()
    features_map = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (ids, images) in enumerate(iter(dataloader)):
            images = images.to(device)
            features = model(images)
            for id_i, f_i in enumerate(zip(ids, features)):
                features_map[id_i] = f_i
        pbar.update()
    json.dump(features_map, open(out_path, 'w'))
    return features_map
