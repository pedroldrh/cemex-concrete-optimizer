# Auditoría de sobrediseño — resultados

**Método (sin modelos, sin IA):** para cada familia, la media exigida por la norma EN 206 es `f_ck + 1.48 × σ` con la desviación estándar σ calculada de los propios resultados de rotura a 28 días. El *exceso* es la media real menos esa exigencia. Cada MPa de exceso se traduce a ~6 kg de cemento por m³ (precio usado: 110 €/t; CO₂: 0.7 kg por kg de cemento).

| Familia | n | f_ck | σ | Media exigida | Media real | Exceso (MPa) | Cemento de más (kg/m³) | Ahorro (€/año) | CO₂ (t/año) |
|---|---|---|---|---|---|---|---|---|---|
| HA-25/B/20/IIa | 90 | 25 | 2.77 | 29.1 | 33.0 | +3.9 | 23 | 46,097 | 293.3 |
| HA-30/B/20/IIa | 80 | 30 | 3.45 | 35.1 | 35.1 | +0.0 | 0 | 413 | 2.6 |
| HA-40/F/12/IIIa | 40 | 40 | 4.00 | 45.9 | 44.0 | -1.9 | 0 | 0 | 0.0 |

**Notas por familia:**

- **HA-25/B/20/IIa**: Exceso relevante sobre el margen exigido por la norma.
- **HA-30/B/20/IIa**: Familia afinada: el margen real está cerca del exigido.
- **HA-40/F/12/IIIa**: ATENCIÓN: la media está POR DEBAJO de la exigida por la norma — riesgo de no conformidad, no hay nada que recortar aquí.

---
**Límites de este análisis.** (1) El recorte real de cualquier familia exige respetar los suelos de durabilidad (cemento mínimo y a/c máxima de su clase de exposición) y validarse con amasadas de prueba; esta hoja no autoriza ningún cambio. (2) Parte de σ puede ser ruido del propio ensayo (curado, prensa, técnico): reducirlo baja la exigencia de la norma sin tocar el hormigón. (3) La conversión MPa→cemento es una regla aproximada; la curva real de cada familia la conoce el equipo de calidad. Decisión final: control de calidad.