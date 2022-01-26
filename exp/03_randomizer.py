import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.randomization import Randomizer
from utils.execute import Executor


amplitude_rate_range = [0.9, 0.95, 1.0, 1.05, 1.1]
frequency_rate_range = [0.9, 0.95, 1.0, 1.05, 1.1]

env_id = 'oscillator-v0'
algo_id = 'ppo'
algo_para = [{}]
train_timestep = int(10e6)
test_timestep = 5000

te_param = {'amplitude_rate': 0.5, 'frequency_rate': 0.5}
dr_set = [{'amplitude_rate': 1.1, 'frequency_rate': 0.95},
          {'amplitude_rate': 1.1, 'frequency_rate': 0.95},
          {'amplitude_rate': 1.1, 'frequency_rate': 0.95},
          {'amplitude_rate': 1.1, 'frequency_rate': 0.95},
          {'amplitude_rate': 0.9, 'frequency_rate': 1.1},
          {'amplitude_rate': 0.9, 'frequency_rate': 1.1},
          {'amplitude_rate': 0.9, 'frequency_rate': 1.1},
          {'amplitude_rate': 0.9, 'frequency_rate': 1.1}]

if __name__ == "__main__":
    #randomizer = Randomizer(amplitude_rate_range, frequency_rate_range)
    #randomizer.random_execute(env_id, algo_id, algo_para[0], train_timestep, test_timestep)

    executor = Executor(env_id, te_param, algo_id, algo_para[0], randomization=dr_set)
    executor.train_policy(train_timestep, save=True)
    executor.test_model(test_timestep)

