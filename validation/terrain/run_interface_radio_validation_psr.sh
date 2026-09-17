#!/usr/bin/env bash
# Run only on the PSR test host, after compiling the isolated source overlay.
# GPU clients are intentionally sequential: the guard measures the whole card.
set -eo pipefail
source /home/yuhanglu/miniconda/etc/profile.d/conda.sh
conda activate corsika_venv
set -u
c8_stage=${1:?isolated PSR stage}
c8_tag=${2:?new validation tag}
c8_phase=$c8_stage/interface-radio
c8_scripts=$c8_stage/source/validation/terrain
c8_cuda=/home/yuhanglu/21CMA/corsika-21cma-kokkos-beta5/build/toolchain-cuda126
export LD_LIBRARY_PATH="$c8_cuda/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export C8_INTERFACE_TEST_THREADS=256 OMP_NUM_THREADS=256 OMP_PROC_BIND=spread OMP_PLACES=threads OPENBLAS_NUM_THREADS=1
if [[ ${C8_RADIO_REUSE_FIXED_TAG:-} == "" ]]; then
mkdir "$c8_phase/waveforms-$c8_tag"
for c8_mode in openmp cuda; do
  taskset -c 0-255 "$c8_stage/build-$c8_mode/tests/accelerator/testKokkosInterfaceRadio" \
    "$c8_phase/propagation-$c8_mode-$c8_tag.csv" "$c8_phase/native-optical-input.txt" "$c8_phase/native-optical-$c8_mode-$c8_tag.csv"
  for c8_kind in uniform plane mesh_shadow mesh_transmitted radial; do
    for c8_limit in .025 .00625; do
      taskset -c 0-255 "$c8_stage/build-$c8_mode/tests/accelerator/testKokkosInterfaceRadioWaveforms" \
        "$c8_phase/waveforms-$c8_tag/${c8_mode}_${c8_kind}_p16_l$c8_limit" "$c8_kind" 16 "$c8_limit"
    done
    taskset -c 0-255 "$c8_stage/build-$c8_mode/tests/accelerator/testKokkosInterfaceRadioWaveforms" \
      "$c8_phase/waveforms-$c8_tag/${c8_mode}_${c8_kind}_p12_l.00625" "$c8_kind" 12 .00625
  done
  taskset -c 0-255 ctest --test-dir "$c8_stage/build-$c8_mode" \
    -R '^(InterfaceTransport_|testKokkosInterfaceQueue)' --output-on-failure \
    > "$c8_phase/transport-regression-$c8_mode-$c8_tag.log" 2>&1
done
python "$c8_scripts/validate_interface_radio_optical.py" \
  --config "$c8_phase/app-openmp-v7/photon_up_resident/radio/config.json" \
  --rays "$c8_phase/native-optical-openmp-$c8_tag.csv" "$c8_phase/native-optical-cuda-$c8_tag.csv" \
  --output "$c8_phase/figures-native-optical-$c8_tag" > "$c8_phase/native-optical-$c8_tag.log" 2>&1
python "$c8_scripts/analyze_interface_radio_waveforms.py" \
  --root "$c8_phase/waveforms-$c8_tag" --output "$c8_phase/figures-waveforms-$c8_tag" \
  > "$c8_phase/waveforms-$c8_tag-analysis.log" 2>&1
printf 'PASS fixed-track and native optical validation %s\n' "$c8_tag"
else
  printf 'Reusing unchanged radio core validation %s\n' "$C8_RADIO_REUSE_FIXED_TAG"
fi
# Exercise the expensive real-DEM CUDA transmission before the remaining
# shower matrix, so a geometry performance regression is detected promptly.
for c8_mode in cuda openmp; do
  taskset -c 0-255 python "$c8_scripts/run_interface_radio_acceptance.py" \
    --stage "$c8_stage" --root "$c8_phase/app-$c8_mode-$c8_tag" --backend "$c8_mode" \
    --cases photon_down photon_up electron_rock nue_forced \
    > "$c8_phase/app-$c8_mode-$c8_tag.log" 2>&1
  printf 'PASS full shower matrix %s %s\n' "$c8_mode" "$c8_tag"
  if [[ "$c8_mode" == cuda && -n ${C8_RADIO_BUFFER_REFERENCE:-} ]]; then
    python "$c8_scripts/validate_interface_radio_candidates.py" --buffering \
      --reference "$C8_RADIO_BUFFER_REFERENCE" --candidate "$c8_phase/app-cuda-$c8_tag/acceptance.json" \
      --output "$c8_phase/buffer-equivalence-$c8_tag.json" > "$c8_phase/buffer-equivalence-$c8_tag.log" 2>&1
    printf 'PASS buffered CUDA radio comparison %s\n' "$c8_tag"
  fi
  if [[ "$c8_mode" == openmp && -n ${C8_RADIO_EXHAUSTIVE_REFERENCE:-} ]]; then
    python "$c8_scripts/validate_interface_radio_candidates.py" \
      --reference "$C8_RADIO_EXHAUSTIVE_REFERENCE" --candidate "$c8_phase/app-openmp-$c8_tag/acceptance.json" \
      --output "$c8_phase/candidate-equivalence-$c8_tag.json" > "$c8_phase/candidate-equivalence-$c8_tag.log" 2>&1
    printf 'PASS exhaustive real-DEM radio comparison %s\n' "$c8_tag"
  fi
done
python "$c8_scripts/validate_interface_radio_atmosphere.py" \
  --config "$c8_phase/app-openmp-$c8_tag/photon_up_resident/radio/config.json" \
  --output "$c8_phase/figures-atmosphere-$c8_tag" > "$c8_phase/atmosphere-$c8_tag.log" 2>&1
python "$c8_scripts/validate_interface_radio_optical.py" \
  --config "$c8_phase/app-openmp-$c8_tag/photon_up_resident/radio/config.json" --prepare "$c8_phase/native-optical-input-$c8_tag.txt"
for c8_mode in openmp cuda; do
  taskset -c 0-255 "$c8_stage/build-$c8_mode/tests/accelerator/testKokkosInterfaceRadio" \
    "$c8_phase/propagation-$c8_mode-$c8_tag.csv" "$c8_phase/native-optical-input-$c8_tag.txt" "$c8_phase/native-optical-$c8_mode-$c8_tag.csv"
done
python "$c8_scripts/validate_interface_radio_optical.py" \
  --config "$c8_phase/app-openmp-$c8_tag/photon_up_resident/radio/config.json" \
  --rays "$c8_phase/native-optical-openmp-$c8_tag.csv" "$c8_phase/native-optical-cuda-$c8_tag.csv" \
  --output "$c8_phase/figures-native-optical-$c8_tag" > "$c8_phase/native-optical-$c8_tag.log" 2>&1
python "$c8_scripts/analyze_interface_radio_showers.py" \
  --acceptance "$c8_phase/app-openmp-$c8_tag/acceptance.json" "$c8_phase/app-cuda-$c8_tag/acceptance.json" \
  --output "$c8_phase/figures-showers-$c8_tag" > "$c8_phase/showers-$c8_tag-analysis.log" 2>&1
export LD_LIBRARY_PATH="$c8_stage/cuda-residency-profile/cupti/lib:$LD_LIBRARY_PATH"
taskset -c 0-255 python "$c8_scripts/profile_interface_residency.py" \
  --acceptance "$c8_phase/app-cuda-$c8_tag/acceptance.json" \
  --plugin "$c8_stage/cuda-residency-profile/libInterfaceResidencyTrace.so" \
  --guard "$c8_scripts/run_interface_radio_guard.py" \
  --root "$c8_phase/profile-$c8_tag" --cases photon_up nue_forced --variants resident --timeout 3600 \
  > "$c8_phase/profile-$c8_tag.log" 2>&1
python "$c8_scripts/analyze_interface_radio_residency.py" \
  --profiles "$c8_phase/profile-$c8_tag/profile_acceptance.json" \
  --output "$c8_phase/profile-$c8_tag/radio_residency.json" > "$c8_phase/profile-$c8_tag-analysis.log" 2>&1
printf 'PASS radio CUDA residency and retained shower figures %s\n' "$c8_tag"
