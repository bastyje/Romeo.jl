import matplotlib.pyplot as plt

tensorflow_times = [8.90, 6.12, 6.64, 6.97, 7.02]
flux_times = [26.14, 3.60, 3.85, 3.86, 3.88]
romeo_times = [8.54, 3.26, 3.69, 3.34, 3.01]
epochs = [1, 2, 3, 4, 5]

tensorflow_sum = sum(tensorflow_times)
flux_sum = sum(flux_times)
romeo_sum = sum(romeo_times)

# plot all three lines
plt.plot(epochs, tensorflow_times, label='TensorFlow', color='blue')
plt.plot(epochs, flux_times, label='Flux', color='orange')
plt.plot(epochs, romeo_times, label='Romeo', color='green')

# add a horizontal line for the sum of all times
plt.axhline(y=tensorflow_sum, color='blue', linestyle='--', label='TensorFlow Total')
plt.axhline(y=flux_sum, color='orange', linestyle='--', label='Flux Total')
plt.axhline(y=romeo_sum, color='green', linestyle='--', label='Romeo Total')

# add labels
plt.xlabel('Epochs')
plt.xticks(epochs)
plt.ylabel('Time (s)')
plt.title('Training Time Comparison')

# add a legend
plt.legend()

# show the plot
plt.show()
