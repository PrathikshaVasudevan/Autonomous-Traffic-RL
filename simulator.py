import pygame
import random
import time
import numpy as np
from dqn_agent import DQNAgent

pygame.init()

WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Traffic Simulation")

clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

signals = {
    "N": "RED",
    "S": "RED",
    "E": "GREEN",
    "W": "GREEN"
}

cars = {"N": [], "S": [], "E": [], "W": []}
waiting_time = {"N": 0, "S": 0, "E": 0, "W": 0}

class Car:
    def __init__(self, lane, is_ambulance=False):
        self.lane = lane
        self.waiting = True
        self.is_ambulance = is_ambulance

        if lane == "N":
            self.x = 380
            self.y = 0
        elif lane == "S":
            self.x = 420
            self.y = HEIGHT
        elif lane == "E":
            self.x = WIDTH
            self.y = 380
        elif lane == "W":
            self.x = 0
            self.y = 420

    def move(self):
        if signals[self.lane] == "GREEN":
            self.waiting = False
            if self.lane == "N":
                self.y += 3
            elif self.lane == "S":
                self.y -= 3
            elif self.lane == "E":
                self.x -= 3
            elif self.lane == "W":
                self.x += 3
        else:
            self.waiting = True

    def draw(self):
        color = (0, 0, 255) if self.is_ambulance else BLACK
        pygame.draw.rect(screen, color, (self.x, self.y, 12, 12))

    def crossed(self):
        if self.lane == "N" and self.y > HEIGHT:
            return True
        if self.lane == "S" and self.y < 0:
            return True
        if self.lane == "E" and self.x < 0:
            return True
        if self.lane == "W" and self.x > WIDTH:
            return True
        return False

def spawn_car():
    lane = random.choice(["N", "S", "E", "W"])

    # 5% chance of ambulance
    is_ambulance = random.random() < 0.05

    cars[lane].append(Car(lane, is_ambulance))

def check_ambulance_override():
    for lane in cars:
        for car in cars[lane]:
            if car.is_ambulance:
                return lane
    return None


def draw_intersection():
    screen.fill(WHITE)
    pygame.draw.rect(screen, GRAY, (350, 0, 100, HEIGHT))
    pygame.draw.rect(screen, GRAY, (0, 350, WIDTH, 100))

def draw_signals():
    font = pygame.font.SysFont(None, 24)
    y = 20
    for lane in ["N", "S", "E", "W"]:
        color = GREEN if signals[lane] == "GREEN" else RED
        pygame.draw.circle(screen, color, (50, y), 10)
        text = font.render(lane, True, BLACK)
        screen.blit(text, (70, y - 8))
        y += 30

def set_action(action):
    if action == 0:
        signals["N"] = signals["S"] = "GREEN"
        signals["E"] = signals["W"] = "RED"
    else:
        signals["N"] = signals["S"] = "RED"
        signals["E"] = signals["W"] = "GREEN"

def get_state():
    counts = [len(cars[l]) for l in ["N", "S", "E", "W"]]
    avg_wait = np.mean(list(waiting_time.values()))
    return np.array(counts + [avg_wait], dtype=np.float32)

def get_reward():
    total_wait = sum(waiting_time.values())
    return -total_wait

# RL Setup
state_size = 5
action_size = 2
agent = DQNAgent(state_size, action_size)

spawn_timer = 0
step_count = 0

prev_state = get_state()

running = True
while running:
    clock.tick(60)
    draw_intersection()
    draw_signals()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    spawn_timer += 1
    if spawn_timer > 50:
        spawn_car()
        spawn_timer = 0

    if step_count % 120 == 0:
        emergency_lane = check_ambulance_override()

        if emergency_lane:
            # Force green for ambulance lane
            if emergency_lane in ["N", "S"]:
                set_action(0)
            else:
                set_action(1)
        else:
            action = agent.act(prev_state)
            set_action(action)

    for lane in cars:
        for car in cars[lane][:]:
            car.move()
            car.draw()

            if car.waiting:
                waiting_time[lane] += 1

            if car.crossed():
                cars[lane].remove(car)

    state = get_state()
    reward = get_reward()
    done = False

    agent.remember(prev_state, action, reward, state, done)
    agent.replay(32)

    prev_state = state

    pygame.display.update()
    step_count += 1

pygame.quit()

agent.save("traffic_dqn.pth")
