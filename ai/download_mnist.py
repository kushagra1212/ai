import gzip
import os
import shutil
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'mnist')
os.makedirs(DATA_DIR, exist_ok=True)

URL_BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]


def download_and_extract():
  print(f'Downloading MNIST dataset to {DATA_DIR}...')
  for filename in FILES:
    filepath = os.path.join(DATA_DIR, filename)
    extracted_path = filepath[:-3]  # remove .gz

    if os.path.exists(extracted_path):
      print(f'  [Already Exists] {extracted_path}')
      continue

    url = URL_BASE + filename
    print(f'  Downloading {url} ...')
    urllib.request.urlretrieve(url, filepath)

    print(f'  Extracting {filename} ...')
    with gzip.open(filepath, 'rb') as f_in:
      with open(extracted_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filepath)  # clean up .gz

  print('\nMNIST dataset successfully downloaded and extracted!')
  print(f'Location: {DATA_DIR}')


if __name__ == '__main__':
  download_and_extract()
