# Validierung auf öffentlichen Datensätzen (BOP)

Stand 03.10.2026, `balanced`-Profil, Apple-Silicon-CPU. Alle Zahlen reproduzierbar mit `snappose bop` (siehe unten).

**Aufbau:** Jede Ground-Truth-Instanz ist ein Versuch. Der Prior ist die GT-Pose plus Zufallsfehler (±6 mm / ±3° je
Achse, im Median ~6 mm / ~3°). Das Tool bekommt nur Tiefenbild, Intrinsik und diesen Prior, kein RGB, keinen Detektor.
GT-Rotationen werden auf SO(3) projiziert (TUD-L-GT ist nur auf ~1e-3 orthonormal).

## Ergebnisse

| Datensatz | Versuche | Prior-Fehler (Median) | Ergebnis-Fehler (Median) | ADD(-S) < 0,1 d | Zeit Ø |
|---|---|---|---|---|---|
| **ITODD** (val, Structured Light) | 120 | 6,0 mm / 2,9° | **2,1 mm / 1,6°** | 78 % | 0,6 s |
| LINEMOD (Kinect-Tiefe) | 300 | 5,9 mm / 3,0° | 5,2 mm / 2,3° | 98 % | 1,0 s |
| TUD-L (Kinect-Tiefe) | 90 | 5,9 mm / 2,9° | 3,9 mm / 2,0° | 100 % | 0,7 s |

ITODD, Anteil ≤ 1 / 2 / 5 mm (und gleiche Gradzahl): 21 % / 42 % / 56 %.

## Interpretation

**ITODD (gute Tiefe):** Bei vielen Objekten konvergiert das Tool auf 0,2–1 mm (z. B. Obj. 2, 21, 26, 27), bei anderen
auf 1–3 mm. Bei einem Teil der Objekte scheitert es (Fehler > 10 mm, meist `NOK` oder `UNSICHER`): Obj. 3, 5, 6, 8–11, 25,
28. Vermutete Ursachen (nicht einzeln untersucht): spiegelnde/dünne Teile mit lückenhafter Tiefe, Haufen mit Verdeckung
und der reine Tiefen-Ansatz. Pro Objekt gibt es nur 1–6 Val-Instanzen, die Aussagen pro Objekt sind also statistisch dünn.
Unter den 6 als `OK` bewerteten Versuchen war keiner > 5 mm daneben.

**LINEMOD / TUD-L (Kinect-Tiefe): kein Beleg für Genauigkeit unter ~5 mm möglich.** Startet man exakt auf der GT-Pose,
wandert das Ergebnis trotzdem im Mittel ~5 mm (LM) bzw. ~4 mm (TUD-L) weg. Messung an der GT-Pose: Die Tiefe weicht dort
selbst um median ±2–6 mm (Streuung 4–6 mm) vom CAD-Modell ab, mit wechselndem Vorzeichen je Bild. Die TUD-L-Ergebnisse
erklären die Tiefe gleich gut oder besser als die GT-Pose (Depth-Score je Bild verglichen). Heißt: Sensorrauschen und
GT-Ungenauigkeit liegen in derselben Größenordnung wie der gemessene Fehler. Rotation verbessert sich dort
leicht (3,0° → 2,3°), die Translation kaum. Die hohen ADD-S-Werte sind wenig aussagekräftig, weil auch der Prior die
0,1·d-Schwelle (10–40 mm) meist schon erfüllt.

**Verifikation:** Auf LM gingen 6 von 17 `OK`-Versuchen mit > 5 mm Fehler durch, auf ITODD 0 von 6. Auf Kinect-Daten
ist „> 5 mm“ allerdings schon die Rauschgrenze, die Zahl ist dort nur bedingt aussagekräftig. Die Schwellen
(`verification.*`, `s4.inlier_tau_mm`) sind nicht für diese Sensoren kalibriert (für LM/TUD-L wurde
`s4.inlier_tau_mm` auf 3 bzw. 6 gesetzt).

## Reproduzieren

```bash
bash tools/fetch_bop.sh                              # lädt LM, TUD-L (models + test_bop19) ~2 GB
# ITODD braucht die val-Daten (Test-GT ist nicht öffentlich):
#   itodd_models.zip und itodd_val.zip von https://huggingface.co/datasets/bop-benchmark/itodd nach data/bop/itodd/ entpacken
snappose bop data/bop/tudl --per-object 30 -s s4.inlier_tau_mm=6
snappose bop data/bop/lm   --per-object 20 -s s4.inlier_tau_mm=3
snappose bop data/bop/itodd --split val --per-object 20 -s s4.inlier_tau_mm=2
```

Parameter: `--prior-t/--prior-r` (Prior-Fehler), `--min-visib` (Mindest-Sichtbarkeit, Default 0,7), `--obj`, `--json`.
Symmetrien kommen aus `models_info.json` (diskret + kontinuierlich), ADD-S wird für symmetrische Objekte verwendet.
