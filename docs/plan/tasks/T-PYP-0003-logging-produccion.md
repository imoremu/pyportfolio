---
id: T-PYP-0003
title: "Logging de producción: nivel WARNING, rotación de fichero y de-duplicación de excepciones"
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

# T-PYP-0003: Logging de producción

## 🎯 Objetivo de Negocio
El logging está en **DEBUG en producción**, escribiendo a un fichero compartido (`C:\Users\Public\Log\logfile.log`) que creció hasta **386 MB**. Además, el mecanismo de "research"/reintentos de pydatastudio vuelve a loguear el mismo traceback muchas veces, amplificando el problema. Bajar a WARNING en producción, rotar el fichero de log y de-duplicar el re-log de la misma excepción evita el crecimiento descontrolado del disco y mantiene los logs legibles y útiles para diagnóstico.

## 📍 Contexto Técnico
- Fichero de log compartido: `C:\Users\Public\Log\logfile.log` (386 MB observados).
- El re-log duplicado proviene del ciclo de "research"/reintentos de pydatastudio, que reintenta y vuelca el mismo traceback repetidamente.

## 📋 Criterios de Aceptación (Nivel Máster)
- [ ] **CA-M-1:** En producción el nivel de logging por defecto es **WARNING** (DEBUG solo activable explícitamente).
- [ ] **CA-M-2:** El fichero de log usa un **`RotatingFileHandler`** con límite de tamaño y número de backups acotado.
- [ ] **CA-M-3:** El re-log del ciclo de research/reintentos deja de volcar el mismo traceback repetidamente (una excepción idéntica no se registra múltiples veces).
- [ ] **CA-M-4:** El tamaño del log en el escenario problemático se mantiene acotado tras los cambios.

## 🛠 Tareas de Componente
- _(pendiente de descomposición técnica — configuración de logging del paquete)_
