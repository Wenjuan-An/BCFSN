import os
import re

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader

from .oxford_pets import OxfordPets

"""
template = ['a photo of a person doing {}.']
"""
template = ['a photo of a {}.']

class UCF101(DatasetBase):

    dataset_dir = 'ucf101'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'UCF-101-midframes')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_UCF101.json')

        self.template = template

        # Load the standard train/validation/test split.
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # Load the fixed few-shot support split.
        preprocessed_train = OxfordPets.build_preprocessed_jsonl_path(
            "ucf101", num_shots, seed, "train"
        )
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train, self.dataset_dir)

        preprocessed_val = OxfordPets.build_preprocessed_jsonl_path(
            "ucf101", 16, 1, "val"
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
    
    def read_data(self, cname2lab, text_file):
        text_file = os.path.join(self.dataset_dir, text_file)
        items = []

        with open(text_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')[0] # trainlist: filename, label
                action, filename = line.split('/')
                label = cname2lab[action]

                elements = re.findall('[A-Z][^A-Z]*', action)
                renamed_action = '_'.join(elements)

                filename = filename.replace('.avi', '.jpg')
                impath = os.path.join(self.image_dir, renamed_action, filename)

                item = Datum(
                    impath=impath,
                    label=label,
                    classname=renamed_action
                )
                items.append(item)
        
        return items
