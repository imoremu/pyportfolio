---
id: T-PYP-0002
title: "Rendimiento del cálculo FIFO/IRPF: eliminar O(n²) con pila de lotes en una pasada"
type: infra
weight: 5
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

# T-PYP-0002: Rendimiento del cálculo FIFO/IRPF

## 🎯 Objetivo de Negocio
`FIFOCalculator` e `IrpfEarningsCalculator` recorren los movimientos con `df.iterrows()` y, por cada venta, **re-filtran todo el DataFrame** buscando las compras previas. Esto da una complejidad **O(n²) por grupo (Test, Producto)**. En fondos monetarios con miles de movimientos el cálculo se vuelve muy lento (se observó Power BI "a rastras" y un log de 386 MB asociado al mismo escenario). Refactorizar a una única pasada cronológica con una pila de lotes por ticker devuelve el cálculo a tiempos razonables y hace utilizable el informe en Power BI.

## 📍 Contexto Técnico
- `pyportfolio/calculators/fifo_calculator.py`
- `pyportfolio/calculators/irpf_earnings_calculator.py`

Patrón actual: `iterrows()` + re-filtrado del DataFrame completo por cada venta.
Patrón objetivo: mantener una **pila de lotes por ticker** (dict) y consumirla en una sola pasada cronológica, sin re-filtrar.

> Nota de coordinación: esta tarea y **T-PYP-0001** tocan los mismos módulos. Idealmente el helper FIFO unificado de T-PYP-0001 ya incorpora el modelo de "pila de lotes por ticker", de modo que ambas tareas convergen en la misma estructura de datos.

## 📋 Criterios de Aceptación (Nivel Máster)
- [ ] **CA-M-1:** El cálculo FIFO/IRPF se realiza en una única pasada cronológica manteniendo una pila de lotes por ticker (sin re-filtrar el DataFrame por venta).
- [ ] **CA-M-2:** Los resultados numéricos son idénticos a los actuales (los 85 tests siguen en verde).
- [ ] **CA-M-3:** Se mide el rendimiento antes/después con un dataset grande (miles de movimientos, escenario tipo fondo monetario) y se documenta la mejora.
- [ ] **CA-M-4:** La complejidad por grupo (Test, Producto) deja de ser cuadrática respecto al número de movimientos.

## 🛠 Tareas de Componente
- _(pendiente de descomposición técnica — implementación directa sobre los calculadores indicados)_
