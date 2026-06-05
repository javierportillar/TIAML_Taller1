# Bonus — Análisis t-SNE / UMAP de respuestas reales del agente

## Qué se hizo

Se extrajeron del Postgres (`conversation_messages`, `role='assistant'`,
`route IS NOT NULL`) un total de **191 respuestas reales** del agente.
Cada texto se vectorizó con el **mismo modelo que usa el RAG en producción**
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensiones)
para que el análisis viva en el mismo espacio semántico que la operación
real. Luego se redujo a 2D con dos técnicas:

- **t-SNE** (perplejidad 12, métrica coseno) — preserva estructura local.
- **UMAP** (n_neighbors 15, métrica coseno) — preserva estructura global.

La intención inicial era verificar visualmente que las distintas rutas del
agente generan respuestas semánticamente distintas. El resultado real
matizó esa hipótesis y obligó a una lectura más cuidadosa.

## Distribución de rutas en el corpus

| Ruta del agente | Turnos | % del corpus |
|-----------------|--------|--------------|
| `consultar_datos_contacto` | 118 | 61.8% |
| `buscar_catalogo_productos` | 56 | 29.3% |
| `memory` | 6 | 3.1% |
| `conversation` | 6 | 3.1% |
| `consultar_informacion_corporativa` | 4 | 2.1% |
| `solicitar_supervisor_humano` | 1 | 0.5% |

**Observación importante**: el corpus está **fuertemente desbalanceado**.
La ruta `consultar_datos_contacto` representa el 62% de todas
las respuestas. Esto se debe a que la mayoría del tráfico durante desarrollo
fueron pruebas de la tool estructurada de contacto. Cualquier métrica de
clustering global queda sesgada por esa dominancia.

## Métricas

- **Silhouette score (cosine, espacio 384d original):** **0.167** — estructura débil.
- **Número de clases (rutas distintas):** 6.
- **Respuestas analizadas:** 191.

Lectura del silhouette en el contexto académico:

| Rango | Interpretación |
|-------|----------------|
| > 0.70 | Estructura fuerte. |
| 0.50 – 0.70 | Estructura razonable. |
| 0.25 – 0.50 | Estructura débil pero defendible. |
| < 0.25 | Prácticamente sin estructura. |

El valor obtenido (0.167) cae **en el rango bajo**: los clústeres existen pero con mucho solapamiento; no se debe presentar como evidencia fuerte de separabilidad.

## Interpretación honesta del plot

> Nota metodológica: t-SNE asigna coordenadas arbitrarias (la orientación cambia
> entre runs aunque la semilla sea la misma si cambia el dataset). Por eso esta
> sección habla en términos de **agrupamiento relativo**, no de posiciones
> absolutas en la gráfica.

**`consultar_datos_contacto` (n=118)** —
contraintuitivamente, **es la clase más dispersa**, no la más densa. Sus
puntos se distribuyen ampliamente por la proyección. La explicación: aunque
todas estas respuestas comparten una frase introductoria ("Encontré esta
información estructurada…"), el **contenido cambia mucho** entre turnos:
unos hablan de WhatsApp, otros de redes sociales, otros de cobertura por
ciudad, otros de horarios. El embedding multilingüe captura el cuerpo de
la respuesta más que la frase de molde, así que la dispersión refleja
diversidad temática real dentro de esta ruta.

**`buscar_catalogo_productos` (n=56)** —
es la clase con **agrupamiento visual más claro**. Sus puntos forman una
nube reconocible aunque no perfectamente delimitada. La explicación: estas
respuestas son casi siempre **tablas de precios** con formato muy similar
(producto, precio, categoría). El embedding identifica ese patrón estructural
y los acerca.

**`memory` (n=6)** —
con solo 6 ejemplos no se puede afirmar que forma un clúster estable. Los
puntos están relativamente cerca entre sí, pero el tamaño muestral no
permite conclusión estadística.

**`conversation` (n=6)** —
mismo caveat que `memory`: 6 puntos no es muestra suficiente. Visualmente
los puntos quedan en una zona común pero no es prueba de clustering.

**`consultar_informacion_corporativa` (n=4)** —
solo 4 puntos. Aparecen dispersos sin estructura visible. Sería necesario
mucho más tráfico de preguntas abiertas para evaluar esta ruta.

**`solicitar_supervisor_humano` (n=1)** —
un solo punto. Es la ruta sensible que pasa por HITL. La estadística aquí
es trivial: con un único ejemplo no hay clúster que medir, solo se confirma
que el sistema lo registró correctamente.

## Lo que el plot enseña, dicho sin inflar

1. **El agente sí responde diferente según la ruta**, pero la diferencia
   no se traduce automáticamente en clústeres separados en el espacio
   semántico. Las rutas con respuestas estructuralmente uniformes
   (`buscar_catalogo_productos`, `memory`, `conversation`) se agrupan
   mejor que la ruta dominante (`consultar_datos_contacto`).
2. **La intuición inicial fue equivocada**. Asumí que las respuestas
   determinísticas (las del JSON estructurado) serían las más fáciles
   de agrupar porque "siguen molde". El plot mostró lo contrario: el
   contenido variado dentro de esa ruta pesa más que el molde.
3. **El silhouette bajo es coherente con lo que se ve**. No es un fallo
   del agente: es una limitación del corpus actual (desbalanceado, sesgado
   por pruebas de desarrollo).

## Qué pasaría con más datos reales

Si el sistema saliera a clientes durante un mes el análisis sería más
útil. Las dimensiones interesantes serían:

- Detectar **conversaciones fallidas** como un clúster propio (turnos
  donde el agente cae al fallback cortés porque el LLM no respondió).
- Identificar **picos de quejas** en periodos específicos
  (`solicitar_supervisor_humano` creciendo = indicador operativo).
- Sub-clústeres densos dentro de `consultar_informacion_corporativa`
  indicarían preguntas frecuentes mal resueltas que ameritan entrar al
  JSON estructurado.

## Archivos generados por este script

| Archivo | Para qué sirve |
|---------|----------------|
| `results/tsne_2d_static.png` | Imagen para embeber en el PDF. |
| `results/tsne_2d_interactive.html` | Plotly con hover sobre cada punto (texto + thread). |
| `results/umap_2d_interactive.html` | Versión UMAP — comparación. |
| `results/tsne_analysis.md` | Este archivo. |

## Cómo reproducir

```bash
cd proyecto
python scripts/run_tsne_analysis.py
```

Tiempo aproximado: 30 segundos (vectorización + ambas reducciones + plots).
