# Experimentacion de Prompt Engineering

Este documento explica como se diseñaron, probaron y ajustaron los prompts del sistema RAG para Sándwich Qbano. Su objetivo es evidenciar el componente de Prompt Engineering de la entrega: no solo se escribieron instrucciones, sino que se iteraron para controlar alucinaciones, mejorar el uso del contexto y producir salidas utiles para resumen, FAQ y Q&A.

## Archivos Relacionados

- `src/prompts.py`: define los prompts por defecto y las funciones que construyen los `ChatPromptTemplate` de LangChain.
- `data/config/prompts_config.json`: contiene los prompts activos que se editan desde la interfaz.
- `src/chains.py`: aplica los prompts, selecciona contexto, ejecuta las cadenas de LangChain y activa respuestas deterministicas para preguntas exhaustivas de catalogo.
- `data/processed/knowledge_base.txt`: contexto limpio que se entrega al modelo.
- `data/processed/chunks.json`: fragmentos recuperables para preguntas enfocadas.
- `results/prompt_experiments.csv`: matriz resumida de pruebas de prompt.

## Problema Inicial Detectado

El comportamiento inicial era demasiado generico. Ante preguntas comerciales como “Enlista todos los productos” o “Que promociones hay?”, el modelo podia responder con categorias inventadas, productos no presentes en el scraping o frases como “detalles no proporcionados”. Esto evidenciaba tres problemas:

- El prompt no obligaba a distinguir entre informacion comercial, institucional y documental.
- El prompt no definia una jerarquia de evidencia.
- El prompt no tenia reglas fuertes para preguntas exhaustivas de catalogo, precios y promociones.

## Estrategia de Diseño

Se separaron los prompts por tarea porque cada funcionalidad necesita un criterio distinto.

### Resumen

El resumen debe sintetizar toda la base de conocimiento, pero sin mezclar hechos comerciales con informacion institucional. Por eso el prompt actual exige:

- Usar exclusivamente el contexto recuperado.
- Aplicar jerarquia de evidencia: catalogo web, paginas institucionales, PDF oficial y ficha local solo si aparece en contexto.
- Separar secciones comerciales, institucionales y sostenibilidad.
- Declarar vacios de informacion.
- Conservar nombres de productos y precios cuando existan.

### FAQ

El FAQ debe convertir la base en preguntas utiles para un usuario final. El prompt actual obliga a:

- Cubrir oferta comercial, precios, promociones, canales, informacion corporativa y limites de datos.
- Incluir una etiqueta de evidencia usada por respuesta.
- No responder si el dato no esta soportado.
- Mantener exactamente el numero de preguntas solicitado.

### Q&A

El Q&A es la tarea mas sensible porque el usuario puede preguntar cualquier cosa. El prompt actual usa una estrategia de clasificacion de intencion:

- `catalogo_productos`
- `promociones`
- `detalle_producto`
- `precios`
- `canales_contacto`
- `institucional`
- `sostenibilidad`
- `comparacion`
- `desconocida`

Luego el modelo debe filtrar el contexto no relacionado y responder solo con evidencia de esa intencion. Para productos, precios y promociones, el prompt exige enumerar solo elementos presentes en el contexto y conservar precios exactos.

## Iteraciones Realizadas

| Version | Problema observado | Ajuste de prompt | Resultado esperado |
|---|---|---|---|
| V0 | El resumen se centraba en ensaladas o en un producto aislado. | Se agrego prioridad por oferta comercial completa y categorias visibles. | El resumen distribuye mejor la informacion por categorias. |
| V1 | El modelo repetia informacion y mezclaba sostenibilidad con catalogo. | Se agrego separacion estricta entre comercial, institucional y sostenibilidad. | Menos repeticion y secciones mas claras. |
| V2 | Q&A inventaba productos genericos si el contexto no era suficiente. | Se agrego regla de no inventar productos, precios ni categorias. | Respuestas mas conservadoras y basadas en evidencia. |
| V3 | Preguntas sobre promociones devolvian “no encontre” aunque habia URL configurada. | Se reforzo la prioridad de fuentes comerciales y promociones en preguntas comerciales. | El modelo busca evidencia comercial antes de usar fuentes institucionales. |
| V4 | Preguntas exhaustivas como “todos los productos” seguian siendo riesgosas para el LLM. | Se combino prompt con una ruta deterministica en `src/chains.py` para catalogo completo. | Listados grandes salen desde datos estructurados, no desde generacion libre. |
| V5 | Faltaba evidencia formal para la rubrica. | Se documentaron criterios, versiones, casos de prueba y archivos afectados. | La experimentacion queda demostrable ante el profesor. |

## Ejemplo de Mejora

### Pregunta

`Enlista TODOS LOS PRODUCTOS`

### Riesgo antes del ajuste

El modelo podia generar productos no encontrados en la base, por ejemplo categorias genericas como bebidas, postres o combos no extraidos del scraping.

### Comportamiento esperado despues del ajuste

La respuesta debe listar solo productos detectados en la base procesada. Si la pregunta es exhaustiva, el sistema usa una respuesta deterministica desde los productos estructurados detectados en `knowledge_base.txt`, agrupados por categoria y con precio cuando exista.

## Criterios de Evaluacion Interna

| Criterio | Pregunta de control | Resultado esperado |
|---|---|---|
| Fidelidad al contexto | ¿El dato aparece en `knowledge_base.txt` o `chunks.json`? | Si no aparece, no se responde como hecho. |
| Control de alucinaciones | ¿El modelo inventa productos o precios? | No debe inventar; debe declarar vacios. |
| Relevancia | ¿La respuesta se enfoca en la intencion de la pregunta? | Debe ignorar contexto tangencial. |
| Cobertura comercial | ¿Lista categorias, productos y precios cuando se pregunta por menu? | Debe priorizar catalogo y precios extraidos. |
| Transparencia | ¿Explica limitaciones cuando faltan datos? | Debe indicar informacion parcial o inexistente. |
| Formato | ¿La salida es legible para usuario final? | Tablas para productos, secciones para resumen y FAQ. |

## Prompt Activo de Q&A

El prompt de Q&A actual se basa en estas reglas:

```text
1. Clasificar internamente la intencion de la pregunta.
2. Usar solo contexto recuperado.
3. Priorizar evidencia comercial para preguntas de productos, precios o promociones.
4. No inventar productos, precios, categorias ni beneficios.
5. Declarar "No encontre esa informacion..." si el dato no aparece.
6. Si hay evidencia parcial, listar solo lo encontrado.
7. Omitir campos inexistentes en vez de escribir "detalles no proporcionados".
```

## Relacion con la Rubrica

Este componente aporta directamente a la categoria de Prompt Engineering porque demuestra:

- Diseño diferenciado para las tres tareas: resumen, FAQ y Q&A.
- Instrucciones claras para basarse en contexto y evitar alucinaciones.
- Iteracion documentada con problemas observados y ajustes concretos.
- Integracion entre prompt y arquitectura RAG, especialmente en preguntas comerciales.
- Evidencia reproducible en archivos del repositorio.

## Como Exponerlo

Para explicar esta parte en clase se puede decir:

> “No usamos un unico prompt generico. Diseñamos prompts por tarea. En resumen pedimos separacion por fuentes y vacios; en FAQ pedimos evidencia usada; y en Q&A primero se clasifica la intencion de la pregunta. Tambien detectamos que los listados exhaustivos de productos eran riesgosos para el LLM, entonces combinamos prompt engineering con una regla deterministica que extrae los productos desde la base procesada. Eso reduce alucinaciones y hace que el sistema responda con datos realmente scrapeados.”

