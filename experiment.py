import numpy as np
from dqn_agent import DQNAgent
from analytics import plot_results

def run_simulation(density, mode="static", steps=500):
    # state = [N, S, E, W, avg_wait]
    state = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    waiting_times = []

    agent = DQNAgent(5, 2)
    if mode == "ai":
        try:
            agent.load("traffic_dqn.pth")
        except:
            pass

    for _ in range(steps):
        # simulate arrivals
        arrivals = np.random.poisson(density, 4)
        state[:4] += arrivals

        if mode == "static":
            action = 0 if _ % 20 < 10 else 1
        else:
            action = agent.act(state)

        # clear some cars
        if action == 0:
            state[0] = max(0, state[0] - 2)
            state[1] = max(0, state[1] - 2)
        else:
            state[2] = max(0, state[2] - 2)
            state[3] = max(0, state[3] - 2)

        avg_wait = np.mean(state[:4])
        state[4] = avg_wait
        waiting_times.append(avg_wait)

    return np.mean(waiting_times)

def run_experiments():
    densities = [1, 2, 3, 4, 5]
    static_waits = []
    ai_waits = []

    for d in densities:
        static_waits.append(run_simulation(d, mode="static"))
        ai_waits.append(run_simulation(d, mode="ai"))

    plot_results(densities, static_waits, ai_waits)

if __name__ == "__main__":
    run_experiments()
