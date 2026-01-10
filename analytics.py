import matplotlib.pyplot as plt

def plot_results(densities, static_waits, ai_waits):
    plt.figure()
    plt.plot(densities, static_waits, label="Static Timer")
    plt.plot(densities, ai_waits, label="AI Control")
    plt.xlabel("Traffic Density")
    plt.ylabel("Average Waiting Time")
    plt.title("Traffic Optimization Performance")
    plt.legend()
    plt.savefig("performance_comparison.png")
    plt.show()
