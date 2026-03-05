# Emergent Spatial Locality from Partial Observation

Numerical experiments demonstrating that spatial locality can emerge from the
structure of partial observation in finite quantum systems. A system of N qubits
with random all-to-all Heisenberg couplings has no built-in spatial structure,
yet an observer who accesses only k < N qubits finds that correlations among
the observed qubits decay with an emergent mutual-information distance. The
effective dimensionality of the observer's correlation geometry is lower than
that of the full system, and this reduction strengthens as the observer becomes
more partial (smaller k/N). The effect persists across Hamiltonian symmetry
classes, survives four independent null models, is robust to the choice of
distance metric, holds for excited states, and scales to N=16 (Hilbert space
dimension 65,536).

## Key Results

- **Correlation decay**: Pearson r = -0.67 (N=8, 50 trials) to -0.74 (N=16, 28 qubit pairs)
- **Cross-observer**: Observer A's MI-distance predicts Observer B's correlations (r = -0.74)
- **No symmetry needed**: Random Pauli Hamiltonians (no continuous symmetry) show r = -0.71
- **Not ground-state-specific**: All tested excited states show negative correlation decay
- **System sizes**: N=8 through N=16 (Hilbert space 256 to 65,536)

## Requirements

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7 (for figure generation only)
- PyTorch >= 2.0 with CUDA (optional; enables GPU-accelerated diagonalization for N >= 10)

Install dependencies:

```
pip install -r requirements.txt
```

## Quick Start

Run the five core experiments (N=8, approximately 5 minutes on CPU):

```
python main.py
```

Options:

```
python main.py --n 10         # Larger system (N=10)
python main.py --exp 2        # Run a single experiment (1-5)
python main.py --no-gpu       # Force CPU-only execution
```

## Full Reproduction

Run all experiments reported in the paper:

```
bash reproduce.sh
```

Total runtime is approximately 3 hours on CPU (the N=16 sparse Lanczos
calculation dominates at ~2 hours). Each script writes JSON results to
the `results/` directory.

### Individual scripts

| Script | Description | Output |
|--------|-------------|--------|
| `main.py` | Core experiments 1-5: symmetry breaking, emergent metric, sheaf convergence, scaling, correlation decay | `results/exp[1-5]_*.json` |
| `run_hardened.py` | Experiments 2 and 5 with bootstrap CIs, permutation tests, and shuffled null model (50 trials, N=8,10) | `results/hardened_stats.json` |
| `run_controls.py` | Three control experiments: 1D chain (positive), Haar random (negative), planted partition | `results/control_[A-C].json` |
| `run_observables.py` | Three independent locality observables: conditional MI, tripartite information, entanglement spectrum | `results/observables.json` |
| `run_circularity.py` | Circularity-breaking tests: coupling distance and cross-observer MI distance | `results/circularity_breaking.json` |
| `run_symmetry_breaking.py` | Symmetry-class robustness: Heisenberg (SU(2)), XXZ (U(1)), random Pauli (none) | `results/symmetry_breaking.json` |
| `run_null_models.py` | Four null models: shuffle, eigenvalue-preserving, degree-preserving, random Hamiltonian | `results/null_models.json` |
| `run_distance_robustness.py` | Five MI-to-distance transformations tested for robustness of the main result | `results/distance_robustness.json` |
| `run_scaling.py` | Scaling study at N=10 and N=12 with full statistical analysis | `results/scaling_N10_N12.json` |
| `run_large_n.py` | Large-system experiments at N=14 and N=16 using sparse Lanczos eigensolver | `results/large_n.json` |
| `run_excited_states.py` | Ground + first 4 excited states: tests whether effect is ground-state-specific | `results/excited_states.json` |

### Figures

After running all experiments, generate publication figures:

```
cd figures
python make_figures_v2.py
```

This produces `fig1_core_result.pdf` through `fig8_excited_states.pdf`.

## Results Directory

All outputs are written as JSON to `results/`. Each file contains raw data,
summary statistics, bootstrap confidence intervals, and p-values. Results are
deterministic given the same NumPy/SciPy versions and random seeds (all seeds
are fixed in the scripts).

## Source Layout

```
src/
  quantum.py           - Hamiltonian construction, exact diagonalization,
                         sparse Lanczos, partial trace, entanglement measures
  experiments.py       - Five core experiments and helper functions
  statistics.py        - Bootstrap CI, permutation tests, four null models
  controls.py          - Control experiments (1D chain, Haar random, planted partition)
  observables.py       - CMI, tripartite information, entanglement spectrum
  coupling_distance.py - Coupling-distance metric and cross-observer tests

paper/
  plc_paper.tex        - Full manuscript (PRA format, ~500 lines)

figures/
  make_figures_v2.py   - Publication figure generation
  fig[1-8]_*.pdf       - Pre-generated figures
```

## Paper

"Emergent Spatial Locality from Partial Observation in Finite Quantum Systems"
by Jason Glass (CIPS Corp LLC).

Target: Physical Review A.

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{glass2026emergent,
  title   = {Emergent Spatial Locality from Partial Observation
             in Finite Quantum Systems},
  author  = {Glass, Jason},
  year    = {2026},
  note    = {Computational component developed in collaboration
             with Opus (Anthropic Claude Opus 4)}
}
```
