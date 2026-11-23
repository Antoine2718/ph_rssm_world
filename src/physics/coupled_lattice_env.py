""" Ground-truth physical simulator of a 2D Coupled Nonlinear Mass-Spring-Damper Lattice
(Fermi-Pasta-Ulam-Tsingou type) driven by external control forces u_t.
The phase space is purely translational (q in R^{2 N^2}, p in R^{2 N^2}) 
with coupled quartic spring potentials, harmonic confinement, viscous Rayleigh dissipation, and external actuation ports.

Simulateur physique de référence (ground-truth) d'un réseau bidimensionnel couplé non linéaire Masse-Ressort-Amortisseur
(de type Fermi-Pasta-Ulam-Tsingou) piloté par des forces de contrôle externes u_t.
L'espace des phases est purement translationnel (q dans R^{2 N^2}, p dans R^{2 N^2}) avec potentiels élastiques
quartiques couplés, confinement harmonique, dissipation visqueuse de Rayleigh et ports d'actionnement externes."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np


class CoupledSpringLattice2D:
    """ 2D elastic lattice of N x N point masses connected to nearest neighbors on a Cartesian grid by nonlinear springs with 
    rest length l_0 = 1.0.
    Total Hamiltonian:
    H(q, p) = sum_i ||p_i||^2 / (2 m) + 0.5 * k_anchor * sum_i ||q_i - q_i^eq||^2
             + sum_{(i, j) in E} [ 0.5 * k_linear * (||q_j - q_i|| - l_0)^2 + 0.25 * k_cubic  * (||q_j - q_i|| - l_0)^4 ]

    Réseau élastique 2D de N x N masses ponctuelles reliées à leurs plus proches voisins sur une grille 
    cartésienne par des ressorts non linéaires de longueur à vide l_0 = 1.0. Hamiltonien total :
    H(q, p) = sum_i ||p_i||^2 / (2 m) + 0.5 * k_anchor * sum_i ||q_i - q_i^eq||^2
              + sum_{(i, j) in E} [ 0.5 * k_linear * (||q_j - q_i|| - l_0)^2
              + 0.25 * k_cubic  * (||q_j - q_i|| - l_0)^4 ] """

    def __init__(
        self,
        grid_size: int = 3,
        dt: float = 0.02,
        mass: float = 1.0,
        k_linear: float = 4.0,
        k_cubic: float = 1.5,
        k_anchor: float = 0.5,
        damping: float = 0.06,
        obs_noise_std: float = 0.01,
        partial_observability: bool = False,
        seed: int = 42,
    ) -> None:
        self.grid_size = grid_size
        self.n_nodes = grid_size * grid_size
        self.dim_q = 2 * self.n_nodes
        self.dt = dt
        self.mass = mass
        self.k_linear = k_linear
        self.k_cubic = k_cubic
        self.k_anchor = k_anchor
        self.damping = damping
        self.obs_noise_std = obs_noise_std
        self.partial_observability = partial_observability
        self.rng = np.random.default_rng(seed)

        # Equilibrium reference coordinates on a regular unit-spaced 2D grid.
        # Coordonnées d'équilibre de référence sur une grille 2D régulière de pas unitaire.
        self.eq_positions = self._build_equilibrium_positions(grid_size)
        self.edges = self._build_grid_edges(grid_size)
        self.rest_lengths = np.ones(len(self.edges), dtype=np.float64)

    @staticmethod
    def _build_equilibrium_positions(n: int) -> np.ndarray:
        """ Builds centered equilibrium coordinates in R^{N^2 x 2} so center of mass is at (0, 0).
        Construit les coordonnées d'équilibre centrées dans R^{N^2 x 2} (centre de masse en (0, 0)). """
        coords = []
        offset = 0.5 * (n - 1)
        for r in range(n):
            for c in range(n):
                coords.append([float(c) - offset, float(r) - offset])
        return np.array(coords, dtype=np.float64)

    @staticmethod
    def _build_grid_edges(n: int) -> np.ndarray:
        """ Builds undirected nearest-neighbor edge list for the N x N grid.
        Construit la liste des arêtes non orientées entre plus proches voisins de la grille N x N. """
        edges = []
        for r in range(n):
            for c in range(n):
                idx = r * n + c
                if c + 1 < n:
                    edges.append((idx, idx + 1))
                if r + 1 < n:
                    edges.append((idx, idx + n))
        return np.array(edges, dtype=np.int64)

    @property
    def obs_dim(self) -> int:
        ''' Returns observation vector dimension depending on full vs. partial observability.
        (dim_q) + optional momenta (dim_q) + 4 global structural invariants (center of mass x/y, spatial gyration radius x/y).
        
        Positions (dim_q) + moments optionnels (dim_q) + 4 invariants structurels globaux
        (centre de masse x/y, rayon de giration spatial x/y). '''
      
        base = self.dim_q + 4
        return base if self.partial_observability else base + self.dim_q

    def kinetic_energy(self, p: np.ndarray) -> float:
        # Computes kinetic energy T(p) = ||p||^2 / (2 m).
        return float(0.5 * np.sum(p ** 2) / self.mass)

    def potential_energy(self, q: np.ndarray) -> float:
        # Computes total elastic + anchor potential energy U(q). Here q represents displacements relative to equilibrium positions eq_positions.
        # Calcule l'énergie potentielle totale élastique + confinement U(q). Ici q représente les déplacements relatifs aux positions d'équilibre eq_positions.
        disp = q.reshape(self.n_nodes, 2)
        abs_pos = self.eq_positions + disp

        # Harmonic anchor potential pinning each mass softly to its equilibrium site.
        # Potentiel harmonique d'ancrage rappelant doucement chaque masse vers son site d'équilibre.
        u_anchor = 0.5 * self.k_anchor * float(np.sum(disp ** 2))

        # Nonlinear FPUT spring energy across all grid bonds.
        # Énergie des ressorts non linéaires FPUT sur toutes les liaisons de la grille.
        u_springs = 0.0
        for e_idx, (i, j) in enumerate(self.edges):
            diff = abs_pos[j] - abs_pos[i]
            dist = float(np.linalg.norm(diff)) + 1e-12
            stretch = dist - self.rest_lengths[e_idx]
            u_springs += 0.5 * self.k_linear * (stretch ** 2) + 0.25 * self.k_cubic * (stretch ** 4)

        return u_anchor + u_springs

    def total_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        # Computes exact mechanical Hamiltonian H(q, p) = T(p) + U(q).
        # Calcule l'Hamiltonien mécanique exact H(q, p) = T(p) + U(q).
        return self.kinetic_energy(p) + self.potential_energy(q)

    def conservative_force(self, q: np.ndarray) -> np.ndarray:
        # Computes exact negative gradient of potential energy F_cons(q) = -grad_q U(q).
        # Calcule le gradient négatif exact de l'énergie potentielle F_cons(q) = -grad_q U(q).
      
        disp = q.reshape(self.n_nodes, 2)
        abs_pos = self.eq_positions + disp
        forces = -self.k_anchor * disp

        for e_idx, (i, j) in enumerate(self.edges):
            diff = abs_pos[j] - abs_pos[i]
            dist = float(np.linalg.norm(diff)) + 1e-12
            stretch = dist - self.rest_lengths[e_idx]
            # dU/dr = k_linear * stretch + k_cubic * stretch^3
            force_mag = self.k_linear * stretch + self.k_cubic * (stretch ** 3)
            unit_vec = diff / dist
            f_ij = force_mag * unit_vec
            forces[i] += f_ij
            forces[j] -= f_ij

        return forces.reshape(-1)

    def instantaneous_power_balance(
        self, q: np.ndarray, p: np.ndarray, u: np.ndarray
    ) -> Tuple[float, float, float]:
        """ Computes exact continuous-time Port-Hamiltonian power terms:
            dH/dt = P_ext - P_diss
        where P_diss = (damping / m^2) * ||p||^2 >= 0 and P_ext = (p / m)^T u.
        
        Calcule les termes exacts de puissance Port-Hamiltonienne en temps continu :
        dH/dt = P_ext - P_diss
        où P_diss = (damping / m^2) * ||p||^2 >= 0 et P_ext = (p / m)^T u. """
      
        vel = p / self.mass
        p_diss = float(self.damping * np.sum(vel ** 2))
        p_ext = float(np.dot(vel, u))
        dh_dt = p_ext - p_diss
        return dh_dt, p_diss, p_ext

    def step(self, q: np.ndarray, p: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Advances (q, p) by one time step dt using Velocity-Verlet symplectic splitting
        for the Hamiltonian flow composed with exact exponential viscous damping.
        
        Fait avancer (q, p) d'un pas de temps dt via le splitting symplectique Velocity-Verlet
        pour le flot Hamiltonien composé avec l'amortissement visqueux exponentiel exact."""
      
        dt = self.dt
        m = self.mass

        # Half-step momentum update with conservative force + external control u.
        # Demi-pas de moment avec force conservative + contrôle externe u.
        f0 = self.conservative_force(q) + u
        p_half = p + 0.5 * dt * f0

        # Full-step position update.
        # Pas complet sur la position.
        q_next = q + dt * (p_half / m)

        # Second half-step momentum update.
        # Second demi-pas sur le moment.
        f1 = self.conservative_force(q_next) + u
        p_verlet = p_half + 0.5 * dt * f1

        # Exact analytical integration of linear viscous damping dp/dt = -(gamma / m) p.
        # Intégration analytique exacte de l'amortissement visqueux linéaire dp/dt = -(gamma / m) p.
        decay = math.exp(-self.damping * dt / m)
        p_next = p_verlet * decay
        return q_next, p_next

    def observe(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        # Generates noisy sensor observations from true phase state (q, p).
        # Génère des observations capteurs bruitées à partir de l'état de phase vrai (q, p).
        noisy_q = q + self.rng.normal(0.0, self.obs_noise_std, size=q.shape)
        disp = q.reshape(self.n_nodes, 2)
        com = disp.mean(axis=0)
        gyration = np.std(disp, axis=0)

        if self.partial_observability:
            return np.concatenate([noisy_q, com, gyration]).astype(np.float32)

        noisy_p = p + self.rng.normal(0.0, self.obs_noise_std, size=p.shape)
        return np.concatenate([noisy_q, noisy_p, com, gyration]).astype(np.float32)

    def rollout(
        self,
        horizon: int,
        control_scale: float = 0.55,
        control_correlation: float = 0.85,
        init_disp_std: float = 0.35,
        init_mom_std: float = 0.35,
        q0: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        # Simulates a full trajectory of length `horizon` with smooth AR(1) external actuation.
        # Simule une trajectoire complète de longueur `horizon` avec actionnement externe lisse AR(1).
        q = self.rng.normal(0.0, init_disp_std, size=(self.dim_q,)) if q0 is None else q0.astype(np.float64).copy()
        p = self.rng.normal(0.0, init_mom_std, size=(self.dim_q,)) if p0 is None else p0.astype(np.float64).copy()

        # Generate temporally correlated external control inputs u_t.
        # Génère des entrées de contrôle externes u_t corrélées temporellement.
        u_seq = np.zeros((horizon, self.dim_q), dtype=np.float64)
        if control_scale > 0.0:
            innovations = self.rng.normal(0.0, control_scale, size=(horizon, self.dim_q))
            alpha = control_correlation
            beta = math.sqrt(max(1e-8, 1.0 - alpha ** 2))
            for t in range(horizon):
                u_seq[t] = innovations[t] if t == 0 else alpha * u_seq[t - 1] + beta * innovations[t]

        qs = np.zeros((horizon + 1, self.dim_q), dtype=np.float32)
        ps = np.zeros((horizon + 1, self.dim_q), dtype=np.float32)
        energies = np.zeros((horizon + 1,), dtype=np.float32)
        diss_rates = np.zeros((horizon,), dtype=np.float32)
        ext_powers = np.zeros((horizon,), dtype=np.float32)
        obs = np.zeros((horizon + 1, self.obs_dim), dtype=np.float32)

        qs[0] = q.astype(np.float32)
        ps[0] = p.astype(np.float32)
        energies[0] = self.total_energy(q, p)
        obs[0] = self.observe(q, p)

        for t in range(horizon):
            _, p_diss, p_ext = self.instantaneous_power_balance(q, p, u_seq[t])
            diss_rates[t] = p_diss
            ext_powers[t] = p_ext

            q, p = self.step(q, p, u_seq[t])
            qs[t + 1] = q.astype(np.float32)
            ps[t + 1] = p.astype(np.float32)
            energies[t + 1] = self.total_energy(q, p)
            obs[t + 1] = self.observe(q, p)

        return {
            "q": qs,
            "p": ps,
            "u": u_seq.astype(np.float32),
            "energy": energies,
            "diss_rate": diss_rates,
            "ext_power": ext_powers,
            "obs": obs,
        }
