import os
from dcgan import *
from utils import *
class Trainer():
    def __init__(self, params):
        super().__init__()
        self.save_epoch = params['save_epoch']
        self.num_epochs = params['nepochs'] 
        self.output_path = params['output_path']
    def train(self, params):
        netG = Generator(params).to(params['device'])
        netG.apply(weights_init)
        print(netG)
        netD = Discriminator(params).to(params['device'])
        netD.apply(weights_init)
        print(netD)
        criterion = nn.BCELoss() 

        viz_noise = torch.randn(params['batch_size'], params['z_dim'], 1, 1, device = params['device'])

        optimizerD = optim.Adam(netD.parameters(), lr = params['lr'], betas=(params['beta1'], 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr= params['lr'], betas=(params['beta1'], 0.999))
        
        # setup some varibles
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        
        if params['dataset'] == 'mnist':
            dataloader = get_mnist(params)
        elif params['dataset'] == 'CelebA':
            dataloader = get_CelebA(params)
        for epoch in range(params['nepochs']):
            D_losses = []
            G_losses = []
            for i, data in enumerate(dataloader):
                x_real = data[0].to(params['device'])
                real_label = torch.full((x_real.size(0),), REAL_LABEL, device = params['device'], dtype = torch.float)
                fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device = params['device'], dtype = torch.float)

                # Update D with real data
                netD.zero_grad()
                y_real = netD(x_real)
                loss_D_real = criterion(y_real, real_label)
                loss_D_real.backward()

                # Update D with fake data
                z_noise = torch.randn(x_real.size(0), params['z_dim'], 1, 1, device = params['device'])
                x_fake = netG(z_noise)
                y_fake = netD(x_fake.detach())
                loss_D_fake = criterion(y_fake, fake_label)

                # Store Discriminator loss
                D_train_loss = loss_D_real + loss_D_fake
                D_losses.append(D_train_loss.item())

                loss_D_fake.backward()
                optimizerD.step()

                # Update G with fake data
                netG.zero_grad()
                y_fake_r = netD(x_fake)
                loss_G = criterion(y_fake_r, real_label)

                # Store Generator loss
                G_losses.append(loss_G.item())

                loss_G.backward()
                optimizerG.step()

                if i % 100 == 0:
                    print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                        epoch, i, len(dataloader),
                        loss_D_real.mean().item(),
                        loss_D_fake.mean().item(),
                        loss_G.mean().item()
                    ))
                    # Export the generated images during training.
                    vutils.save_image(x_real, os.path.join(params['output_path'], 'real_samples.png'), normalize=True)
                    with torch.no_grad():
                        viz_sample = netG(viz_noise)
                        vutils.save_image(viz_sample, os.path.join(params['output_path'], 'fake_samples_{}.png'.format(epoch + 1)), normalize=True)
            
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            # Plot loss curve
            plot_loss(train_hist['D_losses'], train_hist['G_losses'], epoch + 1, params['nepochs'], params['output_path'])
            torch.save(netG.state_dict(), os.path.join(params['output_path'], 'netG_{}.pth'.format(epoch)))
            torch.save(netD.state_dict(), os.path.join(params['output_path'], 'netD_{}.pth'.format(epoch)))

        create_gif(config.epoches, args.save_dir)
