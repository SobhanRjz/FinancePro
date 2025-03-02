import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import gym
from stable_baselines3 import PPO
from deap import base, creator, tools, algorithms
import random

# ----------------------------
# Data Preprocessing
# ----------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def create_features_and_labels(df, lookback=60):
    features, labels = [], []
    for i in range(lookback, len(df)):
        features.append(df.iloc[i-lookback:i].values)
        labels.append(df['close'].iloc[i])
    return np.array(features), np.array(labels)

# ----------------------------
# LSTM Price Prediction Model
# ----------------------------
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------------------------
# Reinforcement Learning Environment (Gym)
# ----------------------------
class TradingEnv(gym.Env):
    def __init__(self, prices):
        self.prices = prices
        self.current_step = 0
        self.position = 0  # -1 = short, 0 = no position, 1 = long
        self.balance = 10000
        self.initial_balance = 10000
        self.last_price = self.prices[0]

        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.balance = 10000
        self.last_price = self.prices[0]
        return np.array([self.prices[self.current_step]])

    def step(self, action):
        current_price = self.prices[self.current_step]
        reward = 0

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.last_price = current_price
        elif action == 2:  # Sell
            if self.position == 1:
                profit = (current_price - self.last_price) * 100  # 100 units
                self.balance += profit
                reward = profit
                self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        return np.array([self.prices[self.current_step]]), reward, done, {}

# ----------------------------
# Genetic Algorithm for Hyperparameter Optimization
# ----------------------------
def genetic_optimization(train_data):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    def individual():
        return [random.uniform(0.0001, 0.01),  # learning_rate
                random.randint(32, 256),       # lstm_units
                random.uniform(0.1, 0.5)]      # dropout_rate

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        learning_rate, lstm_units, dropout_rate = ind

        model = Sequential([
            LSTM(int(lstm_units), return_sequences=True, input_shape=(train_data.shape[1], train_data.shape[2])),
            Dropout(dropout_rate),
            LSTM(int(lstm_units), return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

        model.fit(train_data, train_labels, epochs=3, batch_size=32, verbose=0)
        predictions = model.predict(train_data).flatten()
        return -np.mean(np.abs(predictions - train_labels)),  # Negative MAE (maximize fitness)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=10)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=True)

    best_ind = tools.selBest(population, 1)[0]
    return best_ind

# ----------------------------
# Pipeline - Train and Evaluate
# ----------------------------
if __name__ == "__main__":
    # Load and preprocess data
    df = load_and_preprocess_data("crypto_price_data.csv")
    features, labels = create_features_and_labels(df)

    # Split data
    split = int(0.8 * len(features))
    train_data, test_data = features[:split], features[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    # Use Genetic Algorithm to find best LSTM hyperparameters
    best_params = genetic_optimization(train_data)
    print(f"Best Parameters Found: {best_params}")

    # Train optimized LSTM
    best_lr, best_units, best_dropout = best_params
    model = Sequential([
        LSTM(int(best_units), return_sequences=True, input_shape=(train_data.shape[1], train_data.shape[2])),
        Dropout(best_dropout),
        LSTM(int(best_units), return_sequences=False),
        Dropout(best_dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr), loss='mse')
    model.fit(train_data, train_labels, epochs=10, batch_size=32)

    # Price prediction for backtest
    predicted_prices = model.predict(test_data).flatten()

    # Set up RL environment
    env = TradingEnv(predicted_prices)

    # Train PPO Agent
    model_rl = PPO("MlpPolicy", env, verbose=1)
    model_rl.learn(total_timesteps=10000)

    # Backtest trading performance
    obs = env.reset()
    done = False
    total_profit = 0

    while not done:
        action, _ = model_rl.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_profit += reward

    print(f"Total Profit: ${total_profit:.2f}")

