from functools import partial

from data.dataset import SeparationDataset
from data.musdb import get_musdb_folds
from data.utils import crop_targets, random_amplify


model_shapes = {'output_start_frame': 4776, 'output_end_frame': 93185, 'output_frames': 88409, 'input_frames': 97961}


musdb = get_musdb_folds('/home/kiran/Documents/data/musdb18hq')
# If not data augmentation, at least crop targets to fit model output shape
crop_func = partial(crop_targets, shapes=model_shapes)
# Data augmentation function for training
augment_func = partial(random_amplify, shapes=model_shapes, min=0.7, max=1.0)
train_data = SeparationDataset(musdb, "train", ["bass", "drums", "other", "vocals"], 44100, 2, model_shapes, True,
                               'lmdb', audio_transform=augment_func)
test_data = SeparationDataset(musdb, "test", ["bass", "drums", "other", "vocals"], 44100, 2, model_shapes, True,
                                   'lmdb', audio_transform=augment_func)
val_data = SeparationDataset(musdb, "val", ["bass", "drums", "other", "vocals"], 44100, 2, model_shapes, True,
                                  'lmdb', audio_transform=augment_func)
