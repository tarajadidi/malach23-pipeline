
config = {
    #model
    'vae_epoch': 5,
    'vqvae_checkpoint': 'models\\baseline\\checkpoint\\vqvae', #\\vqvae_020.pt',
    'vqvae_checkpoint_file': 'vqvae_005.pt',
    'ps_epoch': 5,
    'pixelsnail_checkpoint': 'models\\baseline\\checkpoint\\pixelsnail-final',#'\\bottom_005.pt',
    'pixelsnail_checkpoint_file': 'bottom_001.pt',
    'number_of_synthesized_sound_per_class': 10,
    #data
    'train_address': 'datasets\\DCASEFoleySoundSynthesisDevSet',
    #training
    'loader_batchsize': 16,
    'vae_lr': 3e-4,
    'vae_batch': 16,
    'ps_lr': 3e-4,
    'ps_batch': 2,
    'ps_hier': 'bottom',
    'ckpt' : None,
    'sched': None,
    'channel': 256,
    'n_res_block': 4,
    'n_res_channel': 4,
    'n_out_res_block': 0,
    'n_cond_res_block': 3,
    'dropout': 0.1,
    'amp': 'O0',
    'hier': 'bottom'
}