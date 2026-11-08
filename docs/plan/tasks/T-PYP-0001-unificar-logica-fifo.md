---
id: T-PYP-0001
title: "Unificar la lógica de consumo de lotes FIFO (DRY)"
type: infra
weight: 4
version: ""
status: backlog
effort_unit: h
estimated_effort: 0
remaining_effort: 0
actual_effort: 0
created_at: 2026-07-12
updated_at: 2026-07-12
original_ref: ""
---

# T-PYP-0001: Unificar la lógica de consumo de lotes FIFO (DRY)

## 🎯 Objetivo de Negocio
La lógica de consumo de lotes FIFO por ticker está **duplicada en tres sitios**, lo que constituye deuda técnica y una fuente real de bugs: ya se desincronizó una vez (bug de "capacidad de bloqueo" que se corrigió en un módulo pero no en el otro). Unificar esa lógica en un único helper compartido reduce el riesgo de regresiones futuras y garantiza que los tres cálculos (ganancia FIFO, rendimiento IRPF y arrastre de coste en traspasos) produzcan resultados coherentes.

## 📍 Contexto Técnico
Duplicación detectada en:
- `pyportfolio/calculators/fifo_calculator.py`
- `pyportfolio/calculators/irpf_earnings_calculator.py`
- `pyportfolio/calculators/transfer_cost_carry.py`

Antecedente: el bug de "capacidad de bloqueo" se arregló en un sitio y no en el otro por culpa de esta duplicación.

## 📋 Criterios de Aceptación (Nivel Máster)
- [ ] **CA-M-1:** Existe un helper compartido de "consumir lotes FIFO por ticker" reutilizado por los tres calculadores.
- [ ] **CA-M-2:** Los tres calculadores dejan de contener su propia copia de la lógica de consumo de lotes.
- [ ] **CA-M-3:** Tests de equivalencia garantizan que el resultado tras la extracción es idéntico al previo (los 85 tests existentes siguen en verde).
- [ ] **CA-M-4:** Un test dedicado cubre el caso de "capacidad de bloqueo" que motivó el bug original, ejercitando el helper único.

## 🛠 Tareas de Componente
- _(pendiente de descomposición técnica — este es un paquete Python monolítico; se implementará directamente sobre los módulos indicados)_
