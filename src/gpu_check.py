import torch


def check_gpu():
    print("Torch version:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU detected")


if __name__ == "__main__":
    check_gpu()