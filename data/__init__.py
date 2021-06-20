import numpy as np
from torch.utils.data import DataLoader, Subset

from data.collate_fns import PaddingCollateFunction, PaddingCollateFunctionTest
from data.fashionIQ import FashionIQDataset, FashionIQTestDataset, FashionIQTestQueryDataset
from language import AbstractBaseVocabulary

DEFAULT_VOCAB_PATHS = {
    **dict.fromkeys(FashionIQDataset.all_codes(), FashionIQDataset.vocab_path())
}


def _random_indices(dataset_length, limit_size):
    return np.random.randint(0, dataset_length, limit_size)


def train_dataset_factory(transforms, config):
    image_transform = transforms['image_transform']
    text_transform = transforms['text_transform']
    dataset_code = config['dataset']
    use_subset = config.get('use_subset', False)

    if FashionIQDataset.code() in dataset_code:
        dataset_clothing_split = dataset_code.split("_")
        if len(dataset_clothing_split) == 1:
            raise ValueError("Please specify clothing type for this dataset: fashionIQ_[dress_type]")
        clothing_type = dataset_clothing_split[1]
        dataset = FashionIQDataset(split='train', clothing_type=clothing_type, img_transform=image_transform,
                                   text_transform=text_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    if use_subset:
        return Subset(dataset, _random_indices(len(dataset), 1000))

    return dataset


def test_dataset_factory(transforms, config, split='val'):
    image_transform = transforms['image_transform']
    text_transform = transforms['text_transform']
    dataset_code = config['dataset']
    use_subset = config.get('use_subset', False)

    if FashionIQDataset.code() in dataset_code:
        dataset_clothing_split = dataset_code.split("_")
        if len(dataset_clothing_split) == 1:
            raise ValueError("Please specify clothing type for this dataset: fashionIQ_[dress_type]")
        clothing_type = dataset_clothing_split[1]
        test_samples_dataset = FashionIQTestDataset(split=split, clothing_type=clothing_type,
                                                    img_transform=image_transform, text_transform=text_transform)
        test_query_dataset = FashionIQTestQueryDataset(split=split, clothing_type=clothing_type,
                                                       img_transform=image_transform, text_transform=text_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    if use_subset:
        return {"samples": Subset(test_samples_dataset, _random_indices(len(test_samples_dataset), 1000)),
                "query": Subset(test_query_dataset, _random_indices(len(test_query_dataset), 1000))}

    return {"samples": test_samples_dataset,
            "query": test_query_dataset}


def train_dataloader_factory(dataset, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = config.get('shuffle', True)
    # TODO: remove this
    drop_last = batch_size == 32

    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn, drop_last=drop_last)


def test_dataloader_factory(datasets, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = False

    return {
        'query': DataLoader(datasets['query'], batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn),
        'samples': DataLoader(datasets['samples'], batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn)
    }


def create_dataloaders(image_transform, text_transform, configs):
    train_dataset = train_dataset_factory(
        transforms={'image_transform': image_transform['train'], 'text_transform': text_transform['train']},
        config=configs)
    test_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform['val']},
        config=configs)
    train_val_datasets = test_dataset_factory(
        transforms={'image_transform': image_transform['val'], 'text_transform': text_transform['val']},
        config=configs, split='train')
    collate_fn = PaddingCollateFunction(padding_idx=AbstractBaseVocabulary.pad_id())
    collate_fn_test = PaddingCollateFunctionTest(padding_idx=AbstractBaseVocabulary.pad_id())
    train_dataloader = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn)
    test_dataloaders = test_dataloader_factory(datasets=test_datasets, config=configs, collate_fn=collate_fn_test)
    train_val_dataloaders = test_dataloader_factory(datasets=train_val_datasets, config=configs,
                                                    collate_fn=collate_fn_test)
    return train_dataloader, test_dataloaders, train_val_dataloaders
