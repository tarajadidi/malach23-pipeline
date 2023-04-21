from models.baseline_runner import ChallengeBaselineModel
from config import config
import os


#to run vqvae
from datasets.baseline_dataloader import get_dataset_filelist
from torch.utils.data import DataLoader
from typing import List
from helpers import audio2mel



if __name__ == '__main__':
    #initialize baseline model
    generative_model = ChallengeBaselineModel(config)

    #load the train set
    train_file_list: List[dict] = get_dataset_filelist()

    train_set = audio2mel.Audio2Mel(
        train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
    )

    vq_train_loader = DataLoader(
        train_set, batch_size=config['loader_batchsize'], num_workers=4, shuffle=True
    )
    print("training set size: " + str(len(train_set)))

    #train all parts
    generative_model.train(vq_train_loader, config)
    
    #generate!
    generative_model.generate(config)
