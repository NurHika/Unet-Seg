
main():
    print("=" * 30)
    print("Main function of trainUnet.")
    print("=" * 30)

    # Seeding.
    seed = config_unet["seed"]
    tf.random.set_seed(seed)

    # Instantiate custom unet model class.
    unet = unetVgg16()

    # Train the model.
    unet.train()

    print()
    print("Getting out of trainUnet.")
    print("-" * 30)
    print()

    return