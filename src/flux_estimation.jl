# Generator network
function build_generator(noise_dim, output_dim, num_hidden_layers)
    layer_sizes = [noise_dim, fill(128, num_hidden_layers)..., output_dim]
    layers = []

    for i in 1:(length(layer_sizes) - 1)
        push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], relu))
    end
    
    return Chain(layers...)
end

# Discriminator network
function build_discriminator(input_dim, num_hidden_layers)
    layer_sizes = [input_dim, fill(256, num_hidden_layers)..., 1]
    layers = []

    for i in 1:(length(layer_sizes) - 1)
        push!(layers, Dense(layer_sizes[i], layer_sizes[i+1], relu))
   


# GAN model
function build_gan(noise_dim, data_dim, num_hidden_layers)
    generator = build_generator(noise_dim, data_dim, num_hidden_layers)
    discriminator = build_discriminator(data_dim, num_hidden_layers)
    return Chain(generator, discriminator)
end

# Define noise and data dimensions
noise_dim = 100
data_dim = 784  # For MNIST-like images (28x28 pixels)

# Create GAN model
gan = build_gan(noise_dim, data_dim, num_hidden_layers)

# Loss functions
function discriminator_loss(real_output, fake_output)
    return -mean(log(real_output) + log(1. - fake_output))
end

function generator_loss(fake_output)
    return -mean(log(fake_output))
end

# Optimizers
optimizer_discriminator = ADAM(0.0002, 0.5)
optimizer_generator = ADAM(0.0002, 0.5)

# Training loop
function train_gan(gan, data, noise_dim, batch_size, epochs)
    for epoch in 1:epochs
        for batch in Iterators.partition(data, batch_size)
            noise = randn(noise_dim, batch_size)
            fake_data = Flux.batch(gan[1])(noise)
            real_data = Flux.batch(rand(batch_size))

            d_loss = discriminator_loss(Flux.batch(gan[2][1])(real_data), Flux.batch(gan[2][1])(fake_data))
            g_loss = generator_loss(Flux.batch(gan[1])(noise))

            Flux.back!(d_loss)
            Flux.back!(g_loss)

            Flux.update!(optimizer_discriminator, gan[2][1], d_loss)
            Flux.update!(optimizer_generator, gan[1], g_loss)
        end
        println("Epoch $epoch: D Loss: $(d_loss.data), G Loss: $(g_loss.data)")
    end
end

# Load your dataset (e.g., MNIST)
# data = load_data()

# Set your desired batch size and training epochs
# batch_size = 64
# epochs = 100

# Train the GAN
# train_gan(gan, data, noise_dim, batch_size, epochs)