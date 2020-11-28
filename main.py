from dcgan import *
if __name__ == "__main__":
    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()    