echo "TPU=${TPU_ACCELERATOR_TYPE:-unknown}"
pip uninstall -y tensorflow tensorflow-text tf-keras tensorflow-hub tensorflow-decision-forests tensorflow-cpu 2>&1 | grep -E "Successfully" | head -5
pip -q install --no-deps keras-hub 2>&1 | tail -1
pip -q install kagglehub regex rich safetensors 2>&1 | tail -1
python -c "import jax, keras, keras_hub; print('jax', jax.__version__, 'keras', keras.__version__, 'keras_hub', keras_hub.__version__, jax.devices())" 2>&1 | tail -2
echo "== Code 11.1 jax_basics.py (TPU)"; python jax_basics.py 2>&1 | tail -6
echo "== Code 11.2 sharding_compare.py jax (TPU, 1 device)"; python sharding_compare.py jax 2>&1 | tail -4
echo "== Code 11.2 sharding_compare.py jax (CPU, 8 simulated devices)"; JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 python sharding_compare.py jax 2>&1 | tail -4
export KERAS_BACKEND=jax
python jax_gemma.py speed 2>&1 | grep -E "^v6e_jax|Error" | tail -3
python jax_gemma.py logits fp32 highest 2>&1 | grep -E "^v6e_jax|Error" | tail -2
python jax_gemma.py logits fp32 default 2>&1 | grep -E "^v6e_jax|Error" | tail -2
python jax_gemma.py logits bf16 default 2>&1 | grep -E "^v6e_jax|Error" | tail -2
python jax_gemma.py compare 2>&1 | tail -2
python -c "import json; d=json.load(open('jax_results.json')); [v.pop('env',None) for v in d.values() if isinstance(v,dict)]; print('JSON', json.dumps(d))"
echo ALLDONE
