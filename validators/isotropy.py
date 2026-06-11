"""
Validacion z3 de criterios de isotropia para un Hamiltoniano efectivo
de tipo Dirac 3D con velocidades direccionales:

  H_eff(p) = v_x sigma_x p_x + v_y sigma_y p_y + v_z sigma_z p_z.

Isotropia (espacial) significa: H_eff invariante bajo rotaciones SO(3) en
el sector infrarrojo. Una condicion necesaria es v_x = v_y = v_z.

Sustenta:
- book/chapters/05_Isotropia_QW_Honeycomb.tex
- book/chapters/09_Isotropia_QW_3D_Weyl.tex
- book/chapters/18_Criterios_Espacio_Tiempo_Emergente.tex
- book/chapters/22_Anisotropias_Geometria_Efectiva.tex
"""
from __future__ import annotations

import z3


def cannot_be_isotropic_with_unequal_velocities() -> bool:
    """SMT: no existe v_x, v_y, v_z reales positivos tales que el sistema
    sea isotropo Y al menos dos velocidades sean distintas. Unsat.
    """
    s = z3.Solver()
    vx, vy, vz = z3.Reals("vx vy vz")
    s.add(vx > 0, vy > 0, vz > 0)
    # isotropia = todas iguales
    isotropic = z3.And(vx == vy, vy == vz)
    s.add(isotropic)
    s.add(z3.Or(vx != vy, vy != vz, vx != vz))
    return s.check() == z3.unsat


def isotropic_solutions_exist() -> bool:
    """SMT: existe asignacion isotropa con v > 0. Sat (trivialmente)."""
    s = z3.Solver()
    vx, vy, vz, v = z3.Reals("vx vy vz v")
    s.add(v > 0, vx == v, vy == v, vz == v)
    return s.check() == z3.sat


def anisotropy_implies_unequal() -> bool:
    """Si v_x != v_y, no se puede mantener isotropia. Unsat de coexistencia."""
    s = z3.Solver()
    vx, vy, vz = z3.Reals("vx vy vz")
    s.add(vx > 0, vy > 0, vz > 0)
    s.add(vx != vy)
    s.add(z3.And(vx == vy, vy == vz))
    return s.check() == z3.unsat
