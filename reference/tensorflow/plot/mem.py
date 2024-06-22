import matplotlib.pyplot as plt

romeo_alloc = [1.566, 1.384, 1.384, 1.384, 1.384]
flux_alloc = [4.035, 2.637, 2.637, 2.637, 2.637]
tensorflow_alloc = [0.33, 0.19, 0.17, 0.09, 0.09]

epochs = [1, 2, 3, 4, 5]

romeo_sum = sum(romeo_alloc)
flux_sum = sum(flux_alloc)
tensorflow_sum = sum(tensorflow_alloc)

plt.plot(epochs, romeo_alloc, label='Romeo', color='green')
plt.plot(epochs, flux_alloc, label='Flux', color='orange')
plt.plot(epochs, tensorflow_alloc, label='TensorFlow', color='blue')

plt.axhline(y=romeo_sum, color='green', linestyle='--', label='Romeo Total')
plt.axhline(y=flux_sum, color='orange', linestyle='--', label='Flux Total')
plt.axhline(y=tensorflow_sum, color='blue', linestyle='--', label='TensorFlow Total')

plt.xlabel('Epochs')
plt.xticks(epochs)
plt.ylabel('Memory (GiB)')
plt.title('Memory Allocation Comparison')

plt.legend()

plt.show()