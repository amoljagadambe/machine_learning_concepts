from os.path import exists, basename
import noisereduce as nr
import librosa


def remove_noise(org_file):
    """
    Remove the noisy data from audio signal here we are considering the first few milliseconds of audio
    as noisy data
    :param org_file: file path of audio
    :return: Tuple of clean data and sample rate

    Note: Need to tweak the noisy_part according to the audio format.
    TODO: use pydub library to catch the noise in data and use that noise filter in here(noisy_part)
    """

    if not exists(org_file):
        print("No such file: {}".format(basename(org_file)))
        return False
    signal, sr_rate = librosa.load(org_file, 16000)
    noisy_part = signal[00000:7000]
    reduced_noise = nr.reduce_noise(audio_clip=signal, noise_clip=noisy_part, verbose=False)
    return reduced_noise, sr_rate
