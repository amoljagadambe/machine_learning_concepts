from pydub import AudioSegment

file_location = "Enter the file path here"


def file_converter(file_path, export_path):
    """
     This function will convert the audio from source to wave file used in NLP. in backend it uses ffmpeg
     audio tool for conversion, for more detailed parameter Refer
     https://github.com/jiaaro/pydub/

    :param export_path: path for exporting the converted file
    :param file_path: path of the audio file
    :return: export the audio file in wave format (signed int16)
    """
    try:
        sound = AudioSegment.from_file(file_path)
        sound.export(export_path, format="wav",
                     parameters=["-f", "pcm_s16le", "-ac", "1", "-ar", "16000"])
        return True
    except Exception as error:
        print(error)
        return False
