# derpk31pro layer-streaming reference runner and KLD harness image: vLLM 0.32's image for torch 2.13 (cu130),
# transformers 5.19, triton 3.7 and safetensors, plus fla-core (flash-linear-attention's kernels, MIT) for Moonshot's
# Kimi K3 reference modeling code (KDA) and pyarrow for the evaluation corpus. vLLM itself is not used at run time.
# --no-deps keeps the image's torch and triton.
#   docker build -t derpk31pro-runner tools/runner/docker
FROM vllm/vllm-openai:v0.32.0-aarch64
RUN pip install --no-cache-dir --no-deps fla-core==0.6.0 && pip install --no-cache-dir pyarrow==21.0.0
