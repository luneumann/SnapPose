# SnapPose

6D-Lage eines bekannten CAD-Objekts in einem RGB-D-Bild bestimmen, ohne Training pro Objekt.
Zeit und Genauigkeit sind per Profil (`fast`, `balanced`, `precise`) und Zeitbudget einstellbar.

> **Prototyp.** Aktuell nur mit grob bekannter Startlage (Prior-Modus). Ein Modus ohne Startlage ist noch nicht umgesetzt.

## Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
snappose gui          # Weboberfläche auf http://127.0.0.1:8780
snappose demo         # Beispiel ohne Kamera und CAD
pytest
```

Auf dem Mac startet `SnapPose starten.command` die Oberfläche per Doppelklick.

## Eigenes Teil

```bash
snappose onboard --object teil --cad teil.stl            # STL, PLY oder OBJ; --scale 1000 bei Metern
snappose match --object teil --depth depth.png --rgb color.png \
    --intrinsics k.json --prior prior.json --out result.json
```

- Tiefenbild: 16-Bit-PNG in mm, auf das Farbbild registriert. Das Farbbild ist optional und hilft bei flachen Teilen.
- `k.json`: `{"fx":..,"fy":..,"cx":..,"cy":..}`
- `prior.json`: `{"T_cam_obj": [[...4x4...]]}` oder `{"translation_mm": [x,y,z], "quaternion_xyzw": [...]}`

```python
from snappose import PoseMatcher

matcher = PoseMatcher(profile="balanced")
matcher.onboard("teil", "teil.stl")
res = matcher.match("teil", depth_mm, K, prior=T_prior, rgb=image)
res.status, res.T_cam_obj, res.confidence      # OK / UNSICHER / NOK
```

## Grenzen

- Genauigkeit hängt stark von der Tiefenqualität ab. Auf Kinect-Daten (LINEMOD, TUD-L) liegt der Fehler bei etwa 5 mm,
  auf Daten mit guter Tiefe (ITODD) bei etwa 1 mm im Median, mit einzelnen Ausfällen.
- Dicht liegende gleiche Teile und dunkle, kontrastarme Teile sind schwierig. Das Tool meldet das meist als `UNSICHER` oder `NOK`.
- Die Schwellen für `OK` sind Startwerte und sollten pro Objekt und Sensor geprüft werden.

Messwerte und Details: [docs/VALIDATION.md](docs/VALIDATION.md)

Lizenz: MIT
