import matplotlib.pyplot as plt

noise_levels = [0.0001, 0.01, 0.1, 1]
colors = ['blue', 'green', 'orange', 'red']
mses = [984.1162480187107, 2307.7442727906086, 105633.03348577273, 8464119.972116673]

plt.figure()
for i, noise in enumerate(noise_levels):
    plt.scatter(noise, mses[i], color=colors[i])
plt.plot(noise_levels, mses, linestyle='-', color='black', alpha=0.5) 
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Noise level')
plt.ylabel('MSE')
plt.title('MSE vs. Noise Level')
plt.legend(noise_levels)
plt.show()
