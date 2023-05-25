from distutils.core import setup
setup(
  name = 'ModularTorch',         # How you named your package folder (MyLib)
  packages = ['modular_torch'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'custom torch wrapper',   # Give a short description about your library
  author = 'Mahesh Patapalli',                   # Type in your name
  author_email = 'mahesh.patapali@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/sirreajohn/ModularTorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/sirreajohn/ModularTorch/archive/refs/tags/v0.1a.tar.gz',    # I explain this later on
  keywords = ['pytorch', 'python', 'deep learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
        "cmake==3.26.3",
        "lit==16.0.1",
        "mpmath==1.3.0",
        "networkx==3.1",
        "numpy==1.24.2",
        "nvidia-cublas-cu11==11.10.3.66",
        "nvidia-cuda-cupti-cu11==11.7.101",
        "nvidia-cuda-nvrtc-cu11==11.7.99",
        "nvidia-cuda-runtime-cu11==11.7.99",
        "nvidia-cudnn-cu11==8.5.0.96",
        "nvidia-cufft-cu11==10.9.0.58",
        "nvidia-curand-cu11==10.2.10.91",
        "nvidia-cusolver-cu11==11.4.0.1",
        "nvidia-cusparse-cu11==11.7.4.91",
        "nvidia-nccl-cu11==2.14.3",
        "nvidia-nvtx-cu11==11.7.91",
        "pandas==2.0.0",
        "Pillow==9.5.0",
        "plotly==5.14.1",
        "pytz==2023.3",
        "sympy==1.11.1",
        "tenacity==8.2.2",
        "torch==2.0.0",
        "torchaudio==2.0.1",
        "torchvision==0.15.1",
        "tqdm==4.65.0",
        "triton==2.0.0",
        "tzdata==2023.3",
        "torchinfo",
        "tensorboard",
        "torch-tb-profiler",
        "gradio",
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
  ],
)