from matplotlib import pyplot as plt

romeo_acc = [92.08, 92.2, 92.56, 92.78, 93.67]
flux_acc = [89.64, 91.88, 93.01, 93.81, 94.32]
tensorflow_acc = [70.23, 89.49, 91.07, 91.99, 92.47]

epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, romeo_acc, label='Romeo', color='green')
plt.plot(epochs, flux_acc, label='Flux', color='orange')
plt.plot(epochs, tensorflow_acc, label='TensorFlow', color='blue')

plt.xlabel('Epochs')
plt.xticks(epochs)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison')

plt.legend()

plt.show()