def initialize_kaggle():
  !pip install -q kaggle
  from google.colab import files
  files.upload()
  !cp kaggle.json ~/.kaggle/
  !chmod 600 ~/.kaggle/kaggle.json
