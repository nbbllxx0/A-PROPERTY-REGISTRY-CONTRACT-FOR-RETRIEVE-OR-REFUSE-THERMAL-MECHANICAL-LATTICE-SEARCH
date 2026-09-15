# P1-A pore definition (written before any D* catalogue solve)

Chosen definition: **complete periodic pore space**.

`metahomog/pore_phase.py` already distinguishes this from an inlet-accessible
labyrinth. `largest_pore_component` keeps the largest periodic 6-connected
component of the void. On sheet cells that splits the void into two
interpenetrating labyrinths; a homogenisation over both is right for a closed
periodic medium and wrong for a sample fed from one face.

This extension claims only that the registry can take another transport
property as declarations. The catalogue atom is therefore the periodic cell
problem on the complete pore mask (the complement of the solid), not an
inlet-selected component. Pilot numbers 0.371 / φ^1.59 / R²=0.93 stay
unpublished until reproduced on this definition and this catalogue.

If a later study wants accessible-only transport, that is a different
declaration, not a silent switch of this one.
