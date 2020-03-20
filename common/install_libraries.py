import sys
import subprocess
import pkg_resources


def main():
    required = {'pandas', 'configparser','nltk', 'easygui', 'selenium', 'demoji', 'pycorenlp','stanfordnlp','wordcloud',
                'matplotlib', 'numpy', 'prince', 'altair', 'googletrans','sklearn', 'ftfy', 'bs4', 'datetime',
                'textblob', 'tqdm','facebook-scraper','twitter-scraper', 'spacy','gender_guesser','spacy_langdetect'}

    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    try:
        if missing:
            python = sys.executable
            subprocess.check_call([python, '-m', 'pip', 'install', 'torch===1.4.0', 'torchvision===0.5.0', '-f',
                                   'https://download.pytorch.org/whl/torch_stable.html'])
            subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
            subprocess.check_call([python, '-m', 'spacy','download','en_core_web_sm'], stdout=subprocess.DEVNULL)
            subprocess.check_call([python, '-m', 'textblob.download_corpora'], stdout=subprocess.DEVNULL)
        print("All prerequisites are Installed...")
    except subprocess.CalledProcessError as e:
        print(e)
        pass

    return 1


if __name__ == '__main__':
    main()