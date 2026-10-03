# SnapPose

Lage (6D-Pose) **eines bekannten CAD-Objekts** in **einem RGB-D-Bild** bestimmen — ohne Training pro Objekt,
nur mit dem CAD-Modell. Zeit und Genauigkeit sind konfigurierbar (Profile + Einzelparameter + Zeitbudget).
Gedacht als schlanke Alternative bzw. Ergänzung zu klassischem Surface-Based Matching.

> **Status: V1 / Prototyp.** Es gibt den **Prior-Modus** („Lage grob bekannt“, z. B. Teil liegt in einer
> Vorrichtung). Der globale Modus (Lage unbekannt), ein Netz-Refiner (MegaPose) und der Dienst-Endpunkt sind
> geplant, siehe [Roadmap](#roadmap). Alle Messwerte unten stammen von **synthetischen** Daten, noch nicht
> von echten Sensoren.

```
CAD ─ onboard ─► Punkte + Normalen (Cache)
RGB-D + K + Prior-Pose
   S0 Tiefenfilter/ROI → S2 Sobol-Hypothesen im Toleranzfenster → S3 Multi-Hypothesen-ICP + Successive Halving
   → S5 Fein-ICP (Point-to-Plane, Tukey) → S4 Tiefen-Scoring → S6 Status OK / UNSICHER / NOK
```

## Schnellstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                    # 20 Tests, ohne Kamera/GPU
snappose demo             # synthetisches Teil, Ergebnis + Overlay in out/demo_overlay.png
snappose bench -n 30      # Profile gegeneinander messen
```

Mit eigenem Teil (STL/PLY/OBJ; STEP vorher z. B. mit FreeCAD zu STL konvertieren):

```bash
snappose onboard --object teil_4711 --cad teil.stl            # --scale 1000 bei CAD in Metern
python tools/capture_realsense.py --out captures/shot         # optional, braucht pyrealsense2
snappose match --object teil_4711 \
    --depth captures/shot/depth.png --rgb captures/shot/color.png \
    --intrinsics captures/shot/intrinsics.json --prior prior.json \
    -p balanced --out result.json --overlay overlay.png
```

`prior.json`: `{"T_cam_obj": [[...4x4...]]}` oder `{"translation_mm": [x,y,z], "quaternion_xyzw": [...]}`.
`intrinsics.json`: `{"fx":..,"fy":..,"cx":..,"cy":..}`. Tiefe: 16-Bit-PNG in mm (`--depth-scale` für andere Einheiten).

## Python-API

```python
from snappose import PoseMatcher

matcher = PoseMatcher(config="configs/balanced.yaml")      # oder profile="fast"
matcher.onboard("teil_4711", "teil.stl")                   # gecached unter objects/
res = matcher.match("teil_4711", depth_mm, K, prior=T_prior,
                    overrides={"s3": {"refine_iters": 3}}, time_budget_ms=400)
res.status, res.T_cam_obj, res.confidence, res.timing_ms   # res.to_dict() = JSON-Format siehe `MatchResult.to_dict()`
```

## Zeit vs. Genauigkeit

Drei Ebenen: **Profil** (`fast`/`balanced`/`precise`, [`config.py`](src/snappose/config.py)) → **Einzelparameter**
(`-s s3.refine_iters=3`, YAML `overrides:`) → **Zeitbudget** (`--budget 300`): Die Pipeline bricht Refinement-Runden
ab, wenn das Budget knapp wird, und liefert das beste bisherige Ergebnis mit `budget_exhausted: true`
(Status dann nie `OK`).

Synthetischer Benchmark (`snappose bench -n 40`, Teil ~80×50×35 mm, Tiefenrauschen 0,3 mm, Prior-Fehler bis
6 mm / 3° je Achse, Apple-Silicon-CPU, einzelner Thread):

| Profil | ≤1 mm/1° | ≤2 mm/2° | ≤5 mm/5° | Median-Fehler | Zeit Ø / P95 |
|---|---|---|---|---|---|
| fast | 68 % | 90 % | 92 % | 0,66 mm / 0,10° | 11 / 14 ms |
| balanced | 80 % | 100 % | 100 % | 0,65 mm / 0,07° | 39 / 45 ms |
| precise | 88 % | 100 % | 100 % | 0,28 mm / 0,06° | 389 / 453 ms |

## Grenzen (ehrlich)

- **Nur Tiefe:** RGB wird in V1 nicht genutzt. Bei glatten Flächen ist die Lage *in der Fläche* nur schwach
  bestimmt; Fehler im Bereich des Inlier-Toleranzbands (`s4.inlier_tau_mm`, 2 mm) werden vom Scoring nicht
  erkannt. Gröbere Fehler (> 5 mm) erkennt die Verifikation zuverlässig als `UNSICHER`/`NOK`, aber
  im `fast`-Profil gingen im Benchmark einzelne 3–6°-Fehler noch als `OK` durch.
- **Verifikationsschwellen** (`verification.*`) sind Startwerte und müssen pro Objekt/Sensor kalibriert werden.
- **Rückseiten-Culling** setzt ein geschlossenes Mesh mit konsistenten Normalen voraus (sonst: kein Culling).
- Das Objekt sollte freigestellt sein; Tischfläche direkt am Teil (< 10 mm) senkt den Scoring-Wert.
- Symmetrien: nur manuell per `symmetries.json` (zyklisch/kontinuierlich), keine Auto-Erkennung.
- Keine Messung auf echten Sensoren, keine GPU-Pfade; `tools/capture_realsense.py` ist ungetestet.

Symmetrie-Datei:

```json
{"symmetries": [{"type": "cyclic", "axis": [0, 0, 1], "order": 4, "origin": [0, 0, 0]}]}
```

## Roadmap

Nächste Schritte: globaler Modus (DINOv2-Templates + Segmentierung),
MegaPose-Refiner als `REFINERS`-Eintrag, BOP-/Eigendaten-Evaluation, Vergleich mit SBM, Zeitoptimierung (FP16/TensorRT),
FastAPI-Dienst.

## Struktur

```
src/snappose/   api, config, onboarding, geometry, budget, registry, synth, viz, cli, bench
  stages/       s0_preprocess, s2_hypotheses, s3_refine, s4_score, s5_icp, s6_verify
configs/        fast|balanced|precise.yaml        tools/   capture_realsense.py
tests/          pytest
```

Lizenz: MIT. Abhängigkeiten: NumPy, SciPy, trimesh, OpenCV, PyYAML.
