import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.execute import Executor

env_id = 'oscillator-v0'
algo_id = ['ppo', 'ppo', 'ppo', 'ppo', 'ppo']
train_timestep = int(10e6)
test_timestep = 5000
env_para = [{'amplitude_rate': 0.9, 'frequency_rate': 0.95},
            {'amplitude_rate': 0.9, 'frequency_rate': 0.95},
            {'amplitude_rate': 1.05, 'frequency_rate': 1.1},
            {'amplitude_rate': 1.1, 'frequency_rate': 0.95},
            {'amplitude_rate': 1.1, 'frequency_rate': 0.95}]
algo_para = [{}, {}, {}, {}, {}]

#grf_tstcon

def mul_test():
    for i in range(len(env_para)):
        executor = Executor(env_id, env_para[i], algo_id[i], algo_para[i], use_multi_env=True)
        executor.train_policy(train_timestep, save=True)
        executor.test_model(test_timestep)


if __name__=="__main__":
    mul_test()
