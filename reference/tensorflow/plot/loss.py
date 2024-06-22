from matplotlib import pyplot as plt

tensorflow_loss = [1.0753, 0.3818, 0.3159, 0.2804, 0.2602]
romeo_loss = [0.4036, 0.4243, 0.4160, 0.4167, 0.3925]
flux_loss = [0.3903, 0.2970, 0.2506, 0.2210, 0.2008]

epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, tensorflow_loss, label='TensorFlow', color='blue')
plt.plot(epochs, romeo_loss, label='Romeo', color='green')
plt.plot(epochs, flux_loss, label='Flux', color='orange')

plt.xlabel('Epochs')
plt.xticks(epochs)
plt.ylabel('Loss')
plt.title('Loss Comparison')

plt.legend()

plt.show()