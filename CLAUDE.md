# SnapPose — Single-Shot 6D Pose Matching (`snappose`)

Bestimmt die Lage (T_cam_obj, mm) **eines bekannten CAD-Objekts** in **einem RGB-D-Frame**, ohne
objektspezifisches Training. Zeit vs. Genauigkeit per Profil (`fast`/`balanced`/`precise`), Einzelparameter
und Zeitbudget steuerbar. Ausführliche Spezifikation: `docs/SPEC.md` (lokal, nicht im öffentlichen Repo).

**Stand V1 (Iteration 1+2 der Spec):** Prior-Modus (Lage grob bekannt), reine Tiefen-Pipeline auf CPU
(Sobol-Hypothesen → Multi-Hypothesen-ICP mit Successive Halving → Scoring → Fein-ICP → Verifikation).
**Noch nicht:** globaler Modus (DINOv2/Templates), MegaPose-Refiner, RGB-Nutzung, Dienst-API.

```
CAD → onboard (Punkte+Normalen, Cache) ─┐
RGB-D + K + Prior → S0 → S2 → S3 (Refine+Prune) → S5 (ICP) → S4 (Score) → S6 (Status) → MatchResult
```

## Konventionen

- Python ≥ 3.10, venv in `.venv/`; Code/Kommentare Englisch, Doku Deutsch
- Einheiten **mm**, Pose immer `T_cam_obj` 4×4 float64, OpenCV-Kamera (x rechts, y unten, z vorwärts)
- Tiefe: HxW float, mm, 0 = ungültig, auf RGB registriert
- Jede Stufe ist ein Modul in `src/snappose/stages/`; Refiner sind per `REFINERS`-Registry austauschbar
  (`@REFINERS.register("name", license=...)`); `license_mode: commercial` blockiert `non-commercial`/`unknown`
- Zufall immer mit festem Seed → deterministische Ergebnisse
- Status-Werte: `OK` / `UNSICHER` / `NOK`; erschöpftes Zeitbudget liefert nie `OK`
- Keine nicht-kommerziellen Bausteine (FoundationPose, nvdiffrast) im Produktpfad

## Befehle

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest                                        # Tests, ohne Kamera/GPU
snappose demo                                 # synthetischer End-to-End-Lauf + Overlay in out/
snappose bench -n 30                          # Profile vergleichen (Erfolgsquote, Fehler, Zeit)
snappose onboard --object teil --cad teil.stl [--scale 1000] [--symmetry sym.json]
snappose match --object teil --depth d.png --intrinsics k.json --prior prior.json -p balanced
```

## Wichtig beim Ändern

- Scoring/Verifikation sind das Sicherheitsnetz gegen falsche `OK`: nach Änderungen `snappose bench` laufen lassen
  und prüfen, dass grobe Fehler (> 5 mm) nicht als `OK` durchgehen
- ICP verwirft Rückseiten-Korrespondenzen (`model.watertight`); ohne das rutscht dünne Geometrie 1 Plattendicke weg
