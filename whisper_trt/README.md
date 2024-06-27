# Thonburian Whisper to TensorRT

This repository contains instructions and scripts for converting Thonburian Whisper to TensorRT format for efficient inference.

## Building TensorRT Image with TensorRT-LLM

To build the TensorRT image for Ada, Ampere (RTX 40xx series, RTX 30xx series) architectures, follow these steps.

```
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull

make -C docker release_build CUDA_ARCHS="86-real;89-real"

```

After building the Docker image, run the docker container, mapping the current folder to the `/workspace` directory inside the container.

```
docker run -it --gpus=all --ipc=host -v /path/to/this/folder:/workspace tensorrt_llm/release:latest
```


## Converting Thonburian Whisper to TensorRT

We provide the script to convert any Thonburian whisper model to TensorRT engine in one go. 
```
./convert_model.sh --model_name biodatlab/whisper-th-large-v3 --output_name whisper_th_large_v3
```

## Inferencing

To run inference with the converted model, you can use the `run.py` script. For example, ->:

`python3 run.py --engine_dir whisper_th_large_v3 --dataset hf-internal-testing/librispeech_asr_dummy --name librispeech_dummy_large_v3 --batch_size=4`
