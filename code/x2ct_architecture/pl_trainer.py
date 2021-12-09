from argparse import Namespace


from gan.py import GAN

args = {
    'batch_size': 32,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'latent_dim': 100
}
hparams = Namespace(**args)

gan_model = GAN(hparams)

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)    
trainer.fit(gan_model)   

