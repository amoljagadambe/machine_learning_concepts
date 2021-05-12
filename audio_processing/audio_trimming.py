import pandas as pd
import numpy as np
import librosa

"""
Refer This video for trimming audio from wave file Or github repo
https://www.youtube.com/watch?v=mUXkj1BKYk0&list=PLpFebHCwnLFcYwSSH70bDlkUFXJ_dOhhb&index=2&ab_channel=SethAdams

https://github.com/seth814/Audio-Classification
"""


def envelope(y, sr_rate, threshold=0.0005):
    """
     To remove the silence part from the audio signal data

    :param y: audio signal data in float format
    :param sr_rate: sampling rate of audio file
    :param threshold: value below this will considered as silence in audio data
    :return: list with True and False notation

    Note: Tweak the threshold value for better result
    """

    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(sr_rate / 20),
                       min_periods=1,
                       center=True).max()

    return [True if mean > threshold else False for mean in y_mean]


if __name__ == "__main__":
    file_path = "Enter the file path here"
    signal, rate = librosa.load(file_path)
    mask = envelope(signal, rate)
    signal = signal[mask]
    librosa.output.write_wav("save_path", signal, rate)
