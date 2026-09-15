# Second-library checks for §§4.4–4.5

Copied into this isolated tree on 2026-09-05 from the existing 30 Aug 2026
runs. Not a new physics campaign. Logs are the citable outputs. Scripts are
provenance; they still expect the handover cell samples, `scikit-fem`, and
(for the STL cubes) TetGen.

| Manuscript claim | Source log |
|---|---|
| TetGen cubes, k +1.43% / +0.67%; E +6.8% / +2.1%; Schwarz P cube 25% below periodic E11, IWP 2.5% | `verify_scikitfem.log` |
| Periodic elastic hex, six cells (Table `tab:openfem`); solid cube E=1; Backus 1.2e-15 | `verify_catalogue_elastic.log` (cubic + self-check); `21-anisotropic-secondlib.log` (f=112, 113, 123) |
| Periodic conduction, four cells, worst 1.4e-13 (Table `tab:openfem-thermal`) | `22-thermal-secondlib.log` |

Scripts (unchanged copies):

- `verify_scikitfem.py` — TetGen tets + scikit-fem P1 on exported STLs
- `verify_catalogue_elastic.py` — periodic elasticity, scikit-fem hex
- `verify_catalogue_thermal.py` — periodic conduction, scikit-fem hex
