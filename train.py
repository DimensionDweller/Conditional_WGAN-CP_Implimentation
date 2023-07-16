def train(generator, discriminator, optimizer_G, optimizer_D, train_loader, device, start_epoch, num_epochs, critic_iterations, lambda_gp, num_classes, z_dim=100, sample_size=8, log_wandb=True, checkpoint_path=None):
    
    if checkpoint_path is not None and log_wandb:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print('Loaded model from checkpoint')

    for epoch in tqdm(range(start_epoch, num_epochs)):
        for i, (real_samples, labels) in enumerate(train_loader):
            # Generate a batch of fake samples
            labels = labels.to(device)
            batch_size = real_samples.shape[0]
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake_samples = generator(noise, labels)
            real_samples = real_samples.to(device)

            # Train the discriminator (critic)
            for _ in range(critic_iterations):
                fake_samples = generator(noise, labels)
                real_preds_real_fake, real_preds_classes = discriminator(real_samples, labels)
                fake_preds_real_fake, _ = discriminator(fake_samples.detach(), labels)
                gradient_penalty = compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data, labels)
                loss_D_real = -torch.mean(real_preds_real_fake) + torch.mean(fake_preds_real_fake) + lambda_gp * gradient_penalty
                loss_D_class = F.cross_entropy(real_preds_classes.view(real_samples.shape[0], -1), labels)
                loss_D = loss_D_real + loss_D_class

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Train the generator
            if i % critic_iterations == 0:
                fake_samples = generator(noise, labels)
                fake_preds_real_fake, _ = discriminator(fake_samples, labels)
                loss_G = -torch.mean(fake_preds_real_fake)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            # Print losses
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {i}/{len(train_loader)} \
                    Loss D: {loss_D.item()}, loss G: {loss_G.item()}")

        if log_wandb:
            # Log the losses to wandb
            wandb.log({
                'Loss D': loss_D.item(),
                'Loss G': loss_G.item(),
            })

            # Generate some fixed noise samples and pass them through the generator
            fixed_noise = torch.randn(sample_size, 100, 1, 1).to(device)
            fixed_labels = torch.LongTensor([i for i in range(num_classes) for _ in range(sample_size // num_classes)]).to(device)
            remaining = sample_size - len(fixed_labels)
            if remaining > 0:
                fixed_labels = torch.cat([fixed_labels, torch.LongTensor([i for i in range(remaining)]).to(device)])
            generated_images = generator(fixed_noise, fixed_labels).detach().cpu()
                # Unnormalize the images
            generated_images = unnormalize(generated_images)

            # Log the generated images to wandb
            wandb.log({
                "generated images": [wandb.Image(image, caption=f"Generated Image {i}") for i, image in enumerate(generated_images)]
            })

            # Log the generated images to wandb
            wandb.log({
                "generated images": [wandb.Image(image, caption=f"Generated Image {i}") for i, image in enumerate(generated_images)]
            })

            # Save the model checkpoint
            checkpoint_path = f'checkpoint_{epoch}.pth'
         
         
            # Save the model checkpoint
        if epoch % 50 == 0:  # Save only every 50 epochs
            checkpoint_path = f'checkpoint_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss_D': loss_D.item(),
                'loss_G': loss_G.item(),
            }, checkpoint_path)

            # Log the model checkpoint to wandb
            if log_wandb:
                wandb.save(checkpoint_path)

    if log_wandb:
        wandb.finish()
