import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.validation import Validator
from utils.read_file import FileReader

amplitude_rate_range = [0.9, 0.95, 1.0, 1.05, 1.1]
frequency_rate_range = [0.9, 0.95, 1.0, 1.05, 1.1]

if __name__ == "__main__":
    #validator = Validator(amplitude_rate_range, frequency_rate_range, manual_define={'train': [], 'test': [21, 12, 4]}, tst_folder='dr')
    #validator.baseline_test()
    #validator.cross_test('few-shot')

    #filereader = FileReader(amplitude_rate_range, frequency_rate_range)
    #filereader.read_zero('095-095')
    #filereader.rearrange()

    #validator = Validator(amplitude_rate_range, frequency_rate_range, manual_define={'train': [21, 4], 'test': [12]}, tst_folder='dr')
    #validator.cross_test()

    validator = Validator(amplitude_rate_range, frequency_rate_range, manual_define={'train': [], 'test': [25, 26]}, tst_folder='domain_rand')
    validator.baseline_test()
