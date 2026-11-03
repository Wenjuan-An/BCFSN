import os
import pickle
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import (
    Datum, DatasetBase, read_json, write_json, load_jsonl, dict2datum,
    build_data_loader
)

"""
template = ['a photo of a {}, a type of pet.']
"""
template = ['a photo of a {}.']

class OxfordPets(DatasetBase):

    dataset_dir = 'oxford_pets'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')

        self.template = template

        # Load the standard train/validation/test split.
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # Load the fixed few-shot support split.
        preprocessed_train = OxfordPets.build_preprocessed_jsonl_path(
            "oxford_pets", num_shots, seed, "train"
        )
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train, self.dataset_dir)

        preprocessed_val = OxfordPets.build_preprocessed_jsonl_path(
            "oxford_pets", 16, 1, "val"
        )
        val = OxfordPets.load_preprocessed_jsonl(preprocessed_val, self.dataset_dir)

        # Base-to-novel uses separate base and novel evaluation classes.
        if setting == "base2new":
            train, val, test_base, test_new = OxfordPets.base2new_split(train, val, test)
        elif setting == "standard":
            train, val, test_base, test_new = train, val, test, None        

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)
        

        print(f"Number of Train, Val and Test classes = {len(self.classnames)}, {len(self.val_classnames)} and {len(self.test_classnames)}")
        if setting == "base2new":
            assert all([ct == cv for ct, cv in zip(self.classnames, self.val_classnames)]) # ensure train and val classes are the same
            assert not any([ct == cv for ct, cv in zip(self.classnames, self.test_new_classnames)]) # ensure train and test classes are different
    
    @staticmethod
    def build_preprocessed_jsonl_path(dataset_name, num_shots, seed, split):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        jsonl_root = os.environ.get(
            "BCFSN_JSONL_ROOT",
            os.path.join(project_root, "dataset", "jsonl_16shot")
        )

        preprocessed = os.path.join(
            jsonl_root,
            dataset_name,
            "split_fewshot",
            f"shot_{num_shots}-seed_{seed}_{split}.jsonl"
        )

        if not os.path.exists(preprocessed):
            raise FileNotFoundError(
                f"Preprocessed few-shot file not found: {preprocessed}"
            )

        return preprocessed
    
    @staticmethod
    def load_preprocessed_jsonl(preprocessed, dataset_dir=None):
        import json

        assert os.path.exists(preprocessed), (
            f"Please make sure to fix the preprocessed subset at {preprocessed}"
        )
        print(f"Loading preprocessed few-shot data in .jsonl format from {preprocessed}")

        def get_first_existing(item, keys, name):
            for key in keys:
                if key in item:
                    return item[key]
            raise KeyError(
                f"Cannot find `{name}` in jsonl item. "
                f"Available keys are: {list(item.keys())}"
            )

        items = []
        with open(preprocessed, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue

                item = json.loads(line)

                impath = get_first_existing(
                    item,
                    ["impath", "_impath", "image", "image_path", "img_path", "filepath", "file_path", "path", "filename"],
                    "image path"
                )

                label = int(get_first_existing(
                    item,
                    ["label", "_label", "target", "class_label"],
                    "label"
                ))

                classname = get_first_existing(
                    item,
                    ["classname", "_classname", "class_name", "category", "name"],
                    "classname"
                )

                # Resolve support-set paths relative to the selected dataset root
                if dataset_dir is not None:
                    dataset_name = os.path.basename(os.path.normpath(dataset_dir))
                    norm_path = os.path.normpath(impath)
                    parts = norm_path.split(os.sep)

                    if dataset_name in parts:
                        idx = parts.index(dataset_name)
                        rel_path = os.path.join(*parts[idx + 1:])
                        impath = os.path.join(dataset_dir, rel_path)
                    elif not os.path.isabs(impath):
                        impath = os.path.join(dataset_dir, impath)

                items.append(Datum(impath=impath, label=label, classname=classname))

        return items
    
    def base2new_split(train, val, test):
        print("Setup for setting base2new")
        train_base, val_base, test_base = OxfordPets.subsample_classes(train, val, test, subsample="base")
        _, _, test_new = OxfordPets.subsample_classes(train, val, test, subsample="new")
        
        return train_base, val_base, test_base, test_new


    def load_preprocessed(preprocessed):
        assert os.path.exists(preprocessed), f"Please make sure to fix the preprocessed subset at {preprocessed}"
        print(f"Loading preprocessed few-shot data from {preprocessed}")
        with open(preprocessed, "rb") as file:
            data = pickle.load(file)
            train, val = data["train"], data.get("val", None) # needed for imagenet
        return train, val

    
    def read_data(self, split_file):
        filepath = os.path.join(self.anno_dir, split_file)
        items = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                imname, label, species, _ = line.split(' ')
                breed = imname.split('_')[:-1]
                breed = '_'.join(breed)
                breed = breed.lower()
                imname += '.jpg'
                impath = os.path.join(self.image_dir, imname)
                label = int(label) - 1 # convert to 0-based index
                item = Datum(
                    impath=impath,
                    label=label,
                    classname=breed
                )
                items.append(item)
        
        return items
    
    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            label = item.label
            tracker[label].append(idx)
        
        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        
        return train, val
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        
        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val,
            'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
    
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out
        
        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test
    

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output