from models.baseline import train_vqvae
from models.baseline.vqvae import VQVAE
from torch import optim
from models.baseline.scheduler import CycleScheduler
import torch
import lmdb
from models.baseline.extract_code import extract
from datasets.baseline_dataloader import get_dataset_filelist
from torch.utils.data import DataLoader
from helpers import audio2mel
from models.baseline.pixelsnail import PixelSNAIL
from models.baseline.pixelsnail import train as snail_trainer
from torch import nn
from models.baseline.inference import DCASE2023FoleySoundSynthesis, BaseLineModel
import os
from datasets.baseline_dataloader import LMDBDataset


class ChallengeBaselineModel():

    def __init__(self, config):
        
        self.vae_epoch = config['vae_epoch']
        self.vqvae_checkpoint = config['vqvae_checkpoint']
        self.pixelsnail_epoch = config['ps_epoch']
        self.pixelsnail_checkpoint = config['pixelsnail_checkpoint']
        self.number_of_synthesized_sound_per_class = config['number_of_synthesized_sound_per_class']


    def train_vqvae(self, train_loader, config):
        device = "cuda"
        model = VQVAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['vae_lr'])
        scheduler = None
        if config['sched'] == "cycle":
            scheduler = CycleScheduler(
                optimizer,
                config['vae_lr'],
                n_iter=len(train_loader) * self.vae_epoch,
                momentum=None,
                warmup_proportion=0.05,
            )

        for i in range(self.vae_epoch):
            train_latent_diff, train_average_loss = train_vqvae.train(
                i, train_loader, model, optimizer, scheduler, device
            )

        torch.save(
            model.state_dict(), self.vqvae_checkpoint + f"/vqvae_{str(i + 1).zfill(3)}.pt"
        )


    def extract_code(self, vq_checkpoint, name = 'vqvae-code', train_loader = None):
        device = 'cuda'
        if(train_loader is None):
            train_file_list = get_dataset_filelist()

            train_set = audio2mel.Audio2Mel(
                train_file_list, 22050 * 4, 1024, 80, 256, 22050, 0, 8000
            )

            train_loader = DataLoader(train_set, batch_size=64, sampler=None, num_workers=2)

        model = VQVAE()
        print(vq_checkpoint)
        model.load_state_dict(torch.load(vq_checkpoint, map_location='cpu'))
        model = model.to(device)
        model.eval()

        map_size = 1024 * 1024 * 1024
        #map_size = 100 * 1024 * 1024 * 1024

        env = lmdb.open(name, map_size=map_size)

        extract(env, train_loader, model, device)

    def train_pixelsnail(self, config, train_loader):
        amp = None
        device = 'cuda'
        torch.cuda.empty_cache()
        ckpt = {}
        start_point = 0

        if config['ckpt'] is not None:
            _, start_point = args.ckpt.split('_')
            start_point = int(start_point[0:-3])

            ckpt = torch.load(config['ckpt'])
            config = ckpt['args']

        if config['hier'] == 'top':
            model = PixelSNAIL(
                [10, 43],
                512,
                256,
                5,
                4,
                4,
                256,
                dropout=0.1,
                n_out_res_block=0,
                cond_res_channel=0,
            )

        elif config['hier'] == 'bottom':
            model = PixelSNAIL(
                [20, 86],
                512,
                config['channel'],
                5,
                4,
                config['n_res_block'],
                config['n_res_channel'],
                # attention=False,
                dropout=config['dropout'],
                n_cond_res_block=config['n_cond_res_block'],
                cond_res_channel=config['n_res_channel'],
            )

        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['ps_lr'])

        if amp is not None:
            model, optimizer = amp.initialize(model, optimizer, opt_level=config['amp'])

        model = nn.DataParallel(model)
        model = model.to(device)

        scheduler = None
        if config['sched'] == 'cycle':
            scheduler = CycleScheduler(
                optimizer, config['ps_lr'], n_iter=len(train_loader) * config['ps_epoch'], momentum=None
            )

        for i in range(start_point, start_point + config['ps_epoch']):
            snail_trainer(i, train_loader, model, optimizer, scheduler, device)

            torch.save(
                {'model': model.module.state_dict(), 'args': config},
                f'models/baseline/checkpoint/pixelsnail-final/{config["hier"]}_{str(i + 1).zfill(3)}.pt',
            )    

    def train(self, vq_dataset, config):
        os.makedirs("models/baseline/checkpoint/vqvae/", exist_ok=True)    
        self.train_vqvae(vq_dataset, config)
        self.vae_chk_add = config['vqvae_checkpoint'] + '/' + config['vqvae_checkpoint_file']
        self.extract_code(self.vae_chk_add, train_loader = vq_dataset)
        os.makedirs("models/baseline/checkpoint/pixelsnail-final", exist_ok=True)
        path='models/baseline/vqvae-code/'
        dataset = LMDBDataset(path)
        loader = DataLoader(
            #dataset, batch_size=config['loader_batchsize'], shuffle=True, num_workers=0, drop_last=True
            dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True
        )
        self.train_pixelsnail(config, loader)
        self.ps_chk_add = config['pixelsnail_checkpoint'] + '/' + config['pixelsnail_checkpoint_file']
        
    def generate(self, config, vq_checkpoint = None, ps_checkpoint = None):
        if(vq_checkpoint is None):
            vq_checkpoint = self.vae_chk_add
        if(ps_checkpoint is None):
            ps_checkpoint = self.ps_chk_add
        dcase_2023_foley_sound_synthesis = DCASE2023FoleySoundSynthesis(
            config['number_of_synthesized_sound_per_class'], config['ps_batch']
        )
        dcase_2023_foley_sound_synthesis.synthesize(
            synthesis_model=BaseLineModel(ps_checkpoint, vq_checkpoint)
        )
