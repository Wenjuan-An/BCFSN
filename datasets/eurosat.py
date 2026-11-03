import os

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader
from .oxford_pets import OxfordPets

"""
template = ['a centered satellite photo of {}.']
"""
template = ['a photo of a {}.']


class EuroSAT(DatasetBase):

    dataset_dir = 'eurosat'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, '2750')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_EuroSAT.json')
        
        self.template = template

        # Load the standard train/validation/test split.
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)


        preprocessed_train = OxfordPets.build_preprocessed_jsonl_path(
            "eurosat", num_shots, seed, "train"
        )
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train, self.dataset_dir)

        preprocessed_val = OxfordPets.build_preprocessed_jsonl_path(
            "eurosat", 16, 1, "val"
        )
        val = OxfordPets.load_preprocessed_jsonl(preprocessed_val, self.dataset_dir)


        # Base-to-novel uses separate base and novel evaluation classes.
        if setting == "base2new":
            train, val, test_base, test_new = OxfordPets.base2new_split(train, val, test)
        elif setting == "standard":
            train, val, test_base, test_new = train, val, test, None        

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)
