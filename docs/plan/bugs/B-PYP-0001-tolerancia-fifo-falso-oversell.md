---
id: B-PYP-0001
title: "Falso 'overselling' por tolerancia absoluta (1e-9) al consumir lotes FIFO"
type: bug
weight: 8
version: ""
status: completed
effort_unit: h
estimated_effort: 1
remaining_effort: 0
actual_effort: 1
created_at: 2026-07-12
updated_at: 2026-07-12
references: [T-PYP-0001]
assets: []
---

# B-PYP-0001: Falso "overselling" por tolerancia absoluta al consumir lotes FIFO

## 🐞 Descripción
El guard anti-sobreventa aborta el cálculo con datos válidos cuando el importe a vender
coincide (dentro del ruido de coma flotante) con el disponible. Ejemplo real:

```
Cannot sell 76.2550 shares of Fondo ING - Monetario at 2026-03-25 12:00:00;
only 76.2550 are available from previous buys.
```

Ambos valores se imprimen `76.2550` (el mensaje redondea a 4 decimales) pero difieren por
más de `1e-9` (p.ej. disponible = `76.25499998`, a vender = `76.25500000`).

## 🔎 Causa Raíz
La comprobación usa una **tolerancia absoluta** de `1e-9`:

```python
if shares_to_consume > total_available_before + 1e-9:
    raise ValueError(...)
```

`1e-9` es demasiado estricta: para saldos de participaciones de magnitud decenas/miles,
el error de coma flotante acumulado tras consumir muchos lotes en una pasada supera
`1e-9`, disparando un falso positivo. No es un problema de `>` vs `>=` (la igualdad ya
está tolerada); es ruido de redondeo no absorbido.

## 📍 Sitios Afectados (lógica duplicada — ver T-PYP-0001)
- `pyportfolio/calculators/fifo_calculator.py:182` (venta — lanza el ValueError)
- `pyportfolio/calculators/irpf_earnings_calculator.py:285` (venta)
- `pyportfolio/calculators/irpf_earnings_calculator.py:419` (transfer-out)

## 💥 Impacto
El cálculo FIFO/IRPF completo aborta con `ValueError` ante datos perfectamente válidos
(especialmente fondos monetarios con muchos movimientos), rompiendo el informe de
Power BI. Bug de producción.

## ✅ Solución (acordada)
Tolerancia **consciente de la magnitud** (relativa + absoluta) más *clamp* al disponible,
aplicada de forma consistente en los tres sitios:

```python
_SHARE_ABS_TOL = 1e-6
_SHARE_REL_TOL = 1e-6
tol = max(_SHARE_ABS_TOL, _SHARE_REL_TOL * total_available_before)
if shares_to_consume > total_available_before + tol:
    raise ...                      # sobreventa GENUINA
shares_to_consume = min(shares_to_consume, total_available_before)  # clamp del ruido
```

Una sobreventa real (vender claramente más de lo disponible) sigue lanzando/registrando.

## 📋 Criterios de Aceptación
- [x] **CA-1:** El caso `76.2550` disponible / `76.2550` a vender (con desviación < tolerancia)
      NO lanza `ValueError` y consume exactamente el disponible (saldo origen → 0).
      → `test_fifo_sell_within_rounding_tolerance_does_not_raise`.
- [x] **CA-2:** Una sobreventa genuina (p.ej. vender 100 con 76 disponibles) SIGUE lanzando
      `ValueError` en la venta FIFO.
      → `test_fifo_genuine_oversell_beyond_tolerance_still_raises` + tests existentes.
- [x] **CA-3:** La misma tolerancia+clamp se aplica en los 3 sitios de forma consistente
      (helper `is_overselling` en `base_calculator.py`).
- [x] **CA-4:** Los 85 tests existentes siguen en verde (87 passed tras añadir los 2 nuevos).

## 🔗 Relación
- Manifestación concreta de la deuda técnica descrita en **T-PYP-0001** (lógica FIFO duplicada).
  La corrección definitiva (single source of truth) llegará al extraer el helper compartido.
