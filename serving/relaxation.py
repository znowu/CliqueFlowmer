"""Optional M3GNet/ASE relaxation, loaded only when explicitly requested."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import matgl
from ase.constraints import UnitCellFilter
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE, LBFGS
from matgl.ext.ase import M3GNetCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def _load_m3gnet_calculator(*, compute_stress: bool = False) -> M3GNetCalculator:
    #
    # Load a PES potential that provides energies, forces (and stress if requested)
    #
    potential = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    return M3GNetCalculator(potential=potential, compute_stress=compute_stress)


def _wrap01(x: np.ndarray) -> np.ndarray:
    #
    # Wrap fractional coordinates to [0, 1)
    #
    y = np.mod(x, 1.0)
    y[y < 0.0] += 1.0
    return y


def _deterministic_nudge_frac(f: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    #
    # Tiny, deterministic, non-random nudge to break exact degeneracies without RNG
    #
    n = f.shape[0]
    if n == 0:
        return f
    bump = (np.arange(n, dtype=np.float64)[:, None] * np.array([1.0, 2.0, 3.0])) * eps
    return _wrap01(f + bump)


def _has_nan_energy_or_forces(atoms) -> bool:
    #
    # Detect NaNs from the calculator and avoid line-search blowups
    #
    try:
        e = atoms.get_potential_energy()
        if not np.isfinite(e):
            return True
    except Exception:
        return True
    try:
        F = atoms.get_forces(apply_constraint=False)
        return not np.isfinite(F).all()
    except Exception:
        return True


def _puff_if_clashing(atoms, *, min_sep_A: float = 0.7, scale: float = 1.01, relax_cell: bool = False) -> None:
    pos = atoms.get_positions()
    if len(pos) < 2:
        return

    # Fast: find any neighbors within min_sep_A (PBC-aware, C-backed)
    # 'ijd' returns i, j, distances; if any exist, we have a clash
    i, j, d = neighbor_list('ijd', atoms, cutoff=min_sep_A - 1e-9)
    has_clash = (len(d) > 0)

    if not has_clash:
        return

    if relax_cell:
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)
    else:
        # Fixed cell: tiny deterministic nudge in fractional coords
        cell = atoms.cell
        frac = np.linalg.solve(cell.T, pos.T).T
        frac = _deterministic_nudge_frac(frac, eps=1e-3)
        atoms.set_positions(np.dot(frac, cell))


def _safe_to_primitive(
    s: Structure,
    *,
    symprec_candidates: Iterable[float] = (5e-3, 2e-3, 1e-3, 5e-4),
    angle_tolerance_candidates: Iterable[float] = (5.0, 2.0, -1.0),
) -> Structure:
    #
    # Try spglib primitive with multiple tolerances; fall back safely if it fails
    #
    # Wrap and nudge before spglib to avoid coincident sites
    f = _wrap01(s.frac_coords)
    f = _deterministic_nudge_frac(f, eps=1e-10)
    s = Structure(s.lattice, s.species, f, coords_are_cartesian=False)

    for sp in symprec_candidates:
        for ang in angle_tolerance_candidates:
            try:
                out = SpacegroupAnalyzer(s, symprec=sp, angle_tolerance=ang).get_primitive_standard_structure()
                if out is not None and np.isfinite(out.lattice.volume) and out.lattice.volume > 1e-6:
                    return out
            except Exception:
                pass

    #
    # Fallbacks that avoid heavy spglib calls
    #
    try:
        out = s.get_primitive_structure()
        if out is not None and out.lattice.volume > 1e-6:
            return out
    except Exception:
        pass

    try:
        out = s.get_reduced_structure(reduction_algo="niggli")
        if out is not None and out.lattice.volume > 1e-6:
            return out
    except Exception:
        pass

    #
    # Ultimate fallback: return wrapped+nudged input
    #
    return s


def refine_to_primitive_fast_strong(
    struct: Structure,
    *,
    target_fmax: float = 0.02,
    max_steps_pos: int = 1100,
    max_steps_cell: int = 350,
    do_cell_relax: bool = True,
    cell_trigger_force: float = 0.12,
    cell_trigger_strain: float = 0.08,
    min_dist: float = 0.5,
    calculator=None,
) -> Structure:

    #
    # Internal calculator cache
    #
    if not hasattr(refine_to_primitive_fast_strong, "_calc_pos"):
        refine_to_primitive_fast_strong._calc_pos = None
    if not hasattr(refine_to_primitive_fast_strong, "_calc_cell"):
        refine_to_primitive_fast_strong._calc_cell = None

    def _get_calc_pos():
        if calculator is not None:
            return calculator
        if refine_to_primitive_fast_strong._calc_pos is None:
            refine_to_primitive_fast_strong._calc_pos = _load_m3gnet_calculator(compute_stress=False)
        return refine_to_primitive_fast_strong._calc_pos

    def _get_calc_cell():
        if refine_to_primitive_fast_strong._calc_cell is None:
            refine_to_primitive_fast_strong._calc_cell = _load_m3gnet_calculator(compute_stress=True)
        return refine_to_primitive_fast_strong._calc_cell

    #
    # Optional fast validity helpers (safe import)
    #
    _sv = _cv = None
    try:
        from optimization.sun import structurally_valid as _sv, compositionally_valid as _cv
    except Exception:
        try:
            # if you have them elsewhere in your repo
            from data.validation import structurally_valid as _sv, compositionally_valid as _cv
        except Exception:
            _sv = _cv = None

    #
    # Helpers
    #
    def _max_force(atoms) -> float:
        f = atoms.get_forces()
        return float(np.max(np.linalg.norm(f, axis=1)))

    def _cell_is_suspicious(pm_struct) -> bool:
        try:
            s_prim = _safe_to_primitive(pm_struct)
            a = float(pm_struct.lattice.volume)
            b = float(s_prim.lattice.volume)
            if a <= 0.0 or b <= 0.0:
                return True
            return (abs(a - b) / b) > cell_trigger_strain
        except Exception:
            return True

    #
    # 0) Canonicalize early
    #
    s0 = _safe_to_primitive(struct)

    #
    # 1) ASE atoms + positions-only calculator
    #
    atoms = AseAtomsAdaptor.get_atoms(s0)
    atoms.calc = _get_calc_pos()

    #
    # 2) Pre-sanitize
    #
    _puff_if_clashing(atoms, relax_cell=False)
    if _has_nan_energy_or_forces(atoms):
        pos = atoms.get_positions()
        cell = atoms.cell
        frac = np.linalg.solve(cell.T, pos.T).T
        frac = _deterministic_nudge_frac(frac, eps=2e-4)
        atoms.set_positions(np.dot(frac, cell))

    #
    # 2.5) Fast bail-out (only if helpers exist)
    #
    if _sv is not None and _cv is not None:
        pm_now = AseAtomsAdaptor.get_structure(atoms)
        if (not _sv(pm_now, min_dist=min_dist)) or (not _cv(pm_now)):
            return _safe_to_primitive(pm_now)

    #
    # 3) Positions-only relax: FIRE settle -> LBFGS finish
    #
    n_fire0 = min(220, max_steps_pos)
    n_fire1 = min(220, max(0, max_steps_pos - n_fire0))
    n_lbfgs = max(0, max_steps_pos - n_fire0 - n_fire1)

    if n_fire0:
        FIRE(atoms, maxmove=0.06, logfile=None).run(fmax=0.06, steps=n_fire0)
    if n_fire1:
        FIRE(atoms, maxmove=0.035, logfile=None).run(fmax=0.03, steps=n_fire1)

    try:
        mf = _max_force(atoms)
    except Exception:
        mf = float("inf")

    if mf > target_fmax and n_lbfgs:
        LBFGS(atoms, maxstep=0.04, logfile=None).run(fmax=target_fmax, steps=n_lbfgs)

    #
    # 4) Rare, bounded cell relax (gated)
    #
    if do_cell_relax:
        try:
            mf2 = _max_force(atoms)
        except Exception:
            mf2 = float("inf")

        pm_mid = AseAtomsAdaptor.get_structure(atoms)
        suspicious = _cell_is_suspicious(pm_mid)

        if (mf2 > cell_trigger_force) or suspicious:
            atoms.calc = _get_calc_cell()
            obj = UnitCellFilter(atoms)

            n_fire_c = min(160, max_steps_cell)
            n_lbfgs_c = max(0, max_steps_cell - n_fire_c)

            if n_fire_c:
                FIRE(obj, maxmove=0.04, logfile=None).run(fmax=max(3.0 * target_fmax, 0.03), steps=n_fire_c)
            if n_lbfgs_c:
                LBFGS(obj, maxstep=0.04, logfile=None).run(fmax=target_fmax, steps=n_lbfgs_c)

    #
    # 5) Return primitive
    #
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    return _safe_to_primitive(relaxed)
