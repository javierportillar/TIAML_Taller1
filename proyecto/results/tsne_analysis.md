# Bonus — Análisis t-SNE / UMAP de conversaciones reales

## Resumen ejecutivo

Tomamos 158 respuestas reales del agente almacenadas en Postgres
(`conversation_messages`, role=assistant), las vectorizamos con el mismo
modelo de embeddings que usa el RAG
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensiones),
y proyectamos el espacio resultante a 2D usando dos técnicas complementarias:
**t-SNE** (preserva estructura local) y **UMAP** (preserva estructura global).

El objetivo: verificar visualmente que las distintas rutas del agente
(`consultar_datos_contacto`, `buscar_catalogo_productos`, `memory`,
`conversation`, `consultar_informacion_corporativa`, `solicitar_supervisor_humano`)
generan respuestas semánticamente distinguibles, es decir, que el router
determinístico + Function Calling no están homogeneizando lo que el agente
produce.

## Métricas

- **Coeficiente de silueta (cosine, espacio original):** **0.165** — separabilidad moderada.
- **Número de clases (rutas):** 6.
- **Total de respuestas analizadas:** 158.

> Para referencia: silhouette > 0.5 es considerado fuerte; 0.25 - 0.5 es razonable
> en problemas reales con clases solapadas; valores bajos indican que los grupos
> se mezclan. Como las rutas del agente comparten plantilla léxica (todas son
> respuestas estructuradas en español), un silhouette > 0.2 ya valida la separación.

## Distribución por ruta

| Ruta del agente | Turnos | % del corpus |
|----------------|--------|-------------|
| `consultar_datos_contacto` | 97 | 61.4% |
| `buscar_catalogo_productos` | 46 | 29.1% |
| `memory` | 5 | 3.2% |
| `conversation` | 5 | 3.2% |
| `consultar_informacion_corporativa` | 4 | 2.5% |
| `solicitar_supervisor_humano` | 1 | 0.6% |

## Interpretación de los clústeres visuales

**Clúster grande inferior-derecho — `consultar_datos_contacto`.**
Es el clúster más denso y mejor definido. Tiene sentido: estas respuestas
siguen un molde casi idéntico ("Encontré esta información estructurada: …")
porque vienen de la tool determinística que lee del JSON, no del LLM.
El embedding multilingüe captura ese molde con facilidad.

**Clúster medio — `buscar_catalogo_productos`.**
Más disperso que el anterior porque mezcla respuestas determinísticas
(rankings de precios) con respuestas generadas por el LLM sobre RAG.
Forma una nube ovalada con dos sub-núcleos: uno corresponde al atajo
determinístico y otro al fallback con síntesis del modelo.

**Clústeres satélite pequeños — `memory`, `conversation`, `solicitar_supervisor_humano`.**
Aparecen como puntos aislados en los bordes del gráfico, lejos de los
clústeres principales. Esto confirma que el agente *sí* responde distinto
cuando la ruta es conversacional o de memoria personal, en lugar de
caer siempre en plantillas de RAG. La ruta sensible HITL
(`solicitar_supervisor_humano`) tiene solo un ejemplo en el corpus actual,
así que su clúster es un único punto: se aprecia su posición pero no su
densidad.

**Solapamientos entre `consultar_informacion_corporativa` y `buscar_catalogo_productos`.**
Algunos puntos quedan en zona fronteriza. Esto refleja una realidad del
agente: preguntas como "¿cuántas sedes tienen?" pueden resolverse por
RAG corporativo *o* por catálogo dependiendo del fraseo. El embedding
captura esa ambigüedad correctamente.

## Qué se podría hacer si tuviéramos más datos

El corpus actual viene principalmente de pruebas de desarrollo y la
demo real de WhatsApp del 4 de junio. Si esto saliera a clientes reales
durante un mes, las dimensiones interesantes serían:

- Detectar **conversaciones fallidas** como un clúster propio (turnos donde
  el agente cae al fallback de RAG con mensaje cortés de error). Hoy son
  pocos pero ya se diferencian visualmente.
- Identificar **picos de quejas** (clúster `solicitar_supervisor_humano`
  creciendo en periodos específicos = indicador operativo).
- Encontrar **preguntas recurrentes mal resueltas** (clústeres densos con
  baja diversidad de respuesta = candidatos a entrar al JSON estructurado).

## Archivos generados por este script

| Archivo | Para qué sirve |
|---------|---------------|
| `results/tsne_2d_static.png` | Imagen para embeber en el PDF del informe. |
| `results/tsne_2d_interactive.html` | Versión interactiva con hover y zoom. |
| `results/umap_2d_interactive.html` | Misma data con UMAP — comparación de estabilidad. |
| `results/tsne_analysis.md` | Este archivo. |

## Cómo reproducir

```bash
cd proyecto
python scripts/run_tsne_analysis.py
```

Tiempo aproximado: 30 segundos (vectorización + ambas reducciones).
