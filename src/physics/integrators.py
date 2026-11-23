""" Structure-preserving and high-order numerical integrators for Hamiltonian and
Port-Hamiltonian neural dynamical systems, fully differentiable via PyTorch autograd.

Intégrateurs numériques d'ordre élevé et préservant la structure géométrique pour les systèmes dynamiques neuronaux 
Hamiltoniens et Port-Hamiltoniens, entièrement différentiables via l'autograd de PyTorch. """
from __future__ import annotations

from typing import Callable, Tuple
import torch
import torch.nn as nn


class RK4Integrator(nn.Module):
    """ Classic 4th-order Runge-Kutta integrator for general Port-Hamiltonian vector fields:
    dx/dt = f_theta(x, u) = (J(x) - R(x)) grad_x H(x) + g(x) u
    While not strictly symplectic for arbitrary dt, RK4 has O(dt^5) local truncation
    error and handles non-separable Hamiltonians and state-dependent R(x), J(x) smoothly.

    Intégrateur de Runge-Kutta classique d'ordre 4 pour des champs de vecteurs Port-Hamiltoniens généraux :
    dx/dt = f_theta(x, u) = (J(x) - R(x)) grad_x H(x) + g(x) u
    Bien que non strictement symplectique pour un dt arbitraire, RK4 possède une erreur de troncature locale 
    en O(dt^5) et gère sans singularité les Hamiltoniens non séparables ainsi que les matrices J(x) et R(x) dépendantes de l'état. """
    def __init__(self) -> None:
        super().__init__()

    def step(
        self,
        vector_field_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Performs one RK4 integration step from x_t to x_{t+1}.
        Effectue un pas d'intégration RK4 de x_t vers x_{t+1}.
        Args:
            vector_field_fn:
                Callable returning (dx_dt, H(x), dissipation_rate). / Fonction retournant (dx_dt, H(x), taux_de_dissipation).
            x:
                Current phase-space state tensor of shape (batch, 2 * n_dof). / 
                Tenseur d'état courant dans l'espace des phases de forme (batch, 2 * n_dof).
            u:
                External control port input tensor of shape (batch, u_dim). / 
                Tenseur d'entrée du port de contrôle externe de forme (batch, u_dim).
            dt:
                Integration time step. / Pas de temps d'intégration.

        Returns:
            x_next:
                Next state tensor of shape (batch, 2 * n_dof). / Tenseur d'état suivant de forme (batch, 2 * n_dof).
            mean_diss_rate:
                Trapezoidal estimate of instantaneous dissipation rate grad_H^T R grad_H.
                / Estimation trapézoïdale du taux instantané de dissipation grad_H^T R grad_H. """
      
        k1, _, diss1 = vector_field_fn(x, u)
        k2, _, diss2 = vector_field_fn(x + 0.5 * dt * k1, u)
        k3, _, diss3 = vector_field_fn(x + 0.5 * dt * k2, u)
        k4, _, diss4 = vector_field_fn(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        mean_diss_rate = (diss1 + 2.0 * diss2 + 2.0 * diss3 + diss4) / 6.0
        return x_next, mean_diss_rate


class SeparableLeapfrogIntegrator(nn.Module):
    """ Störmer-Verlet / Leapfrog symplectic splitting integrator coupled with exact/exponential
        dissipation splitting (Strang splitting):
             1) Half-step momentum update (conservative + control force)
             2) Full-step position update (kinetic velocity)
             3) Half-step momentum update + linear/diagonal damping decay
         Preserves the canonical symplectic 2-form dq ^ dp when R=0 and u=0, preventing
         secular energy drift over arbitrarily long rollouts.

         Intégrateur symplectique de Störmer-Verlet / Leapfrog couplé à un splitting de Strang
         pour la dissipation :
             1) Demi-pas sur l'impulsion p (forces conservatives + contrôle)
             2) Pas complet sur la position q (vitesse cinétique dH/dp)
             3) Demi-pas sur l'impulsion p + décroissance dissipative
         Préserve exactement la 2-forme symplectique canonique dq ^ dp lorsque R=0 et u=0,
         empêchant toute dérive séculaire de l'énergie sur des horizons arbitrairement longs. """

    def __init__(self, n_dof: int) -> None:
        super().__init__()
        self.n_dof = n_dof

    def step(
        self,
        hamiltonian_fn: Callable[[torch.Tensor], torch.Tensor],
        dissipation_and_port_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Executes a symplectic Leapfrog-Strang step on x = [q, p]. / Exécute un pas symplectique Leapfrog-Strang sur x = [q, p]. 
        q = x[..., : self.n_dof]
        p = x[..., self.n_dof :]

        # Helper to compute partial gradients dH/dq and dH/dp via autograd.
        # Fonction auxiliaire calculant les gradients partiels dH/dq et dH/dp via autograd.
        def _grad_qp(q_in: torch.Tensor, p_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.enable_grad():
                q_req = q_in.detach().requires_grad_(True)
                p_req = p_in.detach().requires_grad_(True)
                state = torch.cat([q_req, p_req], dim=-1)
                h_val = hamiltonian_fn(state).sum()
                grad_q, grad_p = torch.autograd.grad(h_val, (q_req, p_req), create_graph=True)
            return grad_q, grad_p

        # Step 1: First half-kick on momentum p using conservative force -dH/dq and control port.
        # Étape 1 : Premier demi-pas sur le moment p via la force conservative -dH/dq et le port u.
        grad_q0, _ = _grad_qp(q, p)
        _, r_mat0, g_mat0 = dissipation_and_port_fn(x, u)
        ctrl0 = torch.bmm(g_mat0, u.unsqueeze(-1)).squeeze(-1)
        f_p0 = -grad_q0 + ctrl0[..., self.n_dof :]
        p_half = p + 0.5 * dt * f_p0

        # Step 2: Full drift on generalized coordinates q using dH/dp.
        # Étape 2 : Pas complet sur les coordonnées généralisées q via dH/dp.
        _, grad_p_half = _grad_qp(q, p_half)
        q_next = q + dt * (grad_p_half + ctrl0[..., : self.n_dof])

        # Step 3: Second half-kick on momentum p + Port-Hamiltonian dissipative contraction.
        # Étape 3 : Second demi-pas sur p + contraction dissipative Port-Hamiltonienne.
        x_mid = torch.cat([q_next, p_half], dim=-1)
        grad_q1, grad_p1 = _grad_qp(q_next, p_half)
        _, r_mat1, g_mat1 = dissipation_and_port_fn(x_mid, u)
        ctrl1 = torch.bmm(g_mat1, u.unsqueeze(-1)).squeeze(-1)

        grad_h1 = torch.cat([grad_q1, grad_p1], dim=-1)
        diss_vec = torch.bmm(r_mat1, grad_h1.unsqueeze(-1)).squeeze(-1)
        diss_rate = torch.bmm(grad_h1.unsqueeze(1), diss_vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        f_p1 = -grad_q1 + ctrl1[..., self.n_dof :] - diss_vec[..., self.n_dof :]
        p_next = p_half + 0.5 * dt * f_p1

        x_next = torch.cat([q_next, p_next], dim=-1)
        return x_next, diss_rate


class DiscreteGradientMidpointIntegrator(nn.Module):
    """ Implicit Midpoint / Discrete-Gradient Port-Hamiltonian Integrator solved via
         fixed-point Picard iterations.
         By evaluating (J(x_mid) - R(x_mid)) grad_H(x_mid) + g(x_mid) u at
         x_mid = 0.5 * (x_t + x_{t+1}), this scheme satisfies the discrete power balance:
             H(x_{t+1}) - H(x_t) ≈ dt * ( -grad_H^T R grad_H + grad_H^T g u )
         to O(dt^3) accuracy and preserves quadratic invariants identically.

      Intégrateur implicite au point milieu / gradient discret Port-Hamiltonien résolu par
         itérations de point fixe de Picard.
         En évaluant (J(x_mid) - R(x_mid)) grad_H(x_mid) + g(x_mid) u en
         x_mid = 0.5 * (x_t + x_{t+1}), ce schéma satisfait le bilan discret de puissance :
             H(x_{t+1}) - H(x_t) ≈ dt * ( -grad_H^T R grad_H + grad_H^T g u )
         à l'ordre O(dt^3) et préserve exactement les invariants quadratiques. """

    def __init__(self, picard_iters: int = 4) -> None:
        super().__init__()
        self.picard_iters = picard_iters

    def step(
        self,
        vector_field_fn: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Solves x_{t+1} = x_t + dt * f(0.5 * (x_t + x_{t+1}), u) via fixed-point unrolling.
            Résout x_{t+1} = x_t + dt * f(0.5 * (x_t + x_{t+1}), u) par déroulage de point fixe. """
      
        # Predictor initialization using explicit Euler step.
        # Initialisation prédicteur par un pas d'Euler explicite.
        f0, _, diss_rate = vector_field_fn(x, u)
        x_next = x + dt * f0

        # Differentiable Picard fixed-point refinement iterations.
        # Itérations différentiables de raffinement de point fixe de Picard.
        for _ in range(self.picard_iters):
            x_mid = 0.5 * (x + x_next)
            f_mid, _, diss_rate = vector_field_fn(x_mid, u)
            x_next = x + dt * f_mid

        return x_next, diss_rate
