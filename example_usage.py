from client import QuantizedModelVramTensorParallelEstimatorClient

def main():
    client = QuantizedModelVramTensorParallelEstimatorClient()
    res = client.estimate_vram_requirements(32, 'GGUF_Q4_K_M', 1)
    print('Quantized VRAM & TP Estimator: ' + res['vram_estimate_id'])
    print('Weights Size: ' + str(res['base_model_weights_gb']) + ' GB | Per-GPU VRAM: ' + str(res['recommended_vram_per_gpu_gb']) + ' GB')
    print('Projected Speed: ' + str(res['projected_tokens_per_second']) + ' tok/s')
    print('Hardware: ' + res['compatible_gpu_hardware_tiers'][0])
    print('Dossier URL: ' + res['quantization_tradeoff_dossier_url'])

if __name__ == '__main__':
    main()
