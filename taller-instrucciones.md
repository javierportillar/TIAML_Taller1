Actividad del Módulo 1: Creación de la Base de Conocimiento Semántico y Sistema Q&A (Preguntas y Respuestas)

Objetivo:

Diseñar y construir el núcleo de conocimiento para un futuro asistente virtual (chatbot) de una empresa líder del Valle del Cauca. Los estudiantes deberán extraer información pública de la empresa asignada, procesarla, y crear un sistema básico de Preguntas y Respuestas (Q&A) utilizando técnicas de Prompt Engineering. Esta base será el "cerebro" del chatbot que se desarrollará en los siguientes módulos.

Descripción de la Actividad:

En los mismos grupos de 4 personas, a cada equipo se le asignará una empresa del listado de las "500 Empresas más importantes del Valle". La tarea de este primer módulo es construir y validar la capa de conocimiento de la empresa.

Instrucciones Detalladas:

1. Asignación de Empresa y Análisis Inicial (Investigación):

Asignación: Se asignará una empresa del Top 50 a cada grupo (ej. Grupo 1: CELSIA, Grupo 2: TECNOQUÍMICAS, etc.).

Investigación: Los estudiantes deben realizar una investigación exhaustiva de la empresa asignada para identificar sus fuentes de información pública y digital:

Sitio web oficial (secciones "Quiénes somos", "Nuestros productos/servicios", "Sostenibilidad", "Informes de gestión", "Noticias/Blog").

Perfiles en redes sociales (LinkedIn, Instagram, Facebook, etc.).

Apariciones en medios de comunicación locales y nacionales (noticias, entrevistas).

Definición del Alcance: El grupo debe definir qué tipo de preguntas debería poder responder un cliente al interactuar por primera vez con la empresa (ej: horarios, sedes, tipos de productos, historia de la empresa, información de contacto, procesos básicos).

2. Construcción de la Base de Conocimiento Semántico (Knowledge Base):
Extracción de Datos: Utilizando librerías de Python como requests y BeautifulSoup, Selenium, etc. Los estudiantes deberán hacer web scraping del sitio web de su empresa para extraer el contenido textual relevante.
Preprocesamiento y Segmentación (Chunking): El texto extraído debe ser limpiado (eliminar HTML, etc.) y dividido en fragmentos más pequeños y semánticamente coherentes (chunks).
Nota: Este punto es práctico, se debe entregar el repositorio con todos los códigos utilizados y la base de conocimiento proveniente del web scrapping.

3. Construcción del Aplicativo - NO ES UN RAG:
Selección del Modelo y Framework: Los estudiantes decidirán si usar un modelo Open Source a través de Ollama (ej. Gemma 4, Mistral) o un modelo privado vía API de OpenAI (ej. GPT-5, Gemini 3.1 Pro, etc). Deberán utilizar un Framework de LLM (se recomienda LangChain o LlamaIndex) para orquestar el proceso.
Aplicación de Prompt Engineering: Diseñarán un prompt robusto que instruya al LLM para que responda la pregunta del usuario basándose únicamente en el contexto proporcionado. Deberán experimentar con técnicas como zero-shot, el formato del prompt y las instrucciones para evitar alucinaciones.
Tomen todo el texto limpio que extrajeron en el "Punto 2" (tras el web scraping) y consolídenlo en el prompt de sistema. Este prompt será la "memoria" o el "cerebro" completo de su asistente en esta primera fase.

4. Desarrollo de una Interfaz de Prueba:
Para facilitar la prueba y la demostración, los estudiantes desarrollarán una interfaz de usuario web simple utilizando Streamlit o Gradio. Esta interfaz debe tener un campo para escribir una pregunta y un área donde se muestre la respuesta generada por el sistema.

5. Pruebas, Documentación y Presentación:

Pruebas: Realizarán pruebas exhaustivas formulando al menos 20 preguntas distintas (relacionadas al alcance definido en el paso 1) para evaluar la precisión y coherencia de las respuestas.

Documentación (Informe): El informe deberá seguir la estructura solicitada de las secciones:

Descripción del problema: "Necesidad de un canal de comunicación automatizado y preciso para la empresa X".

Planteamiento de la solución: "Creación de un sistema Q&A basado en RAG como núcleo para un futuro chatbot".

Preparación de los datos: Detallar el proceso de scraping, limpieza y chunking.

Modelado: Explicar la elección del modelo de embedding, el LLM que van a utilizar en el modulo 2, la base de datos vectorial y el diseño del prompt.

Resultados: Mostrar ejemplos de preguntas y respuestas, analizar la calidad de las mismas y discutir las limitaciones encontradas.

Presentación: La sustentación será en vivo (15 minutos), donde deberán explicar el proceso y realizar una demostración funcional de su sistema Q&A a través de la interfaz de Streamlit/Gradio. NO HACER PRESENTACIONES DE POWERPOINT

RÚBRICA DE EVALUACIÓN AJUSTADA (Módulo 1)

Análisis de la Empresa y Estrategia de Datos (15%)
Claridad en la investigación de la empresa y definición del alcance del Q&A.
Calidad y relevancia de los datos extraídos mediante scraping.
Procesamiento de Datos y Tareas de LLM (25%)

Calidad de la consolidación del conocimiento en un archivo de texto limpio.
Implementación funcional de las tres tareas solicitadas (Resumen, FAQ, Q&A) usando LangChain.
Calidad del Prompt Engineering (25%)

Eficacia y creatividad en el diseño de los prompts para cada tarea.
Inclusión de instrucciones claras para basarse en el contexto y evitar alucinaciones.
Documentación del proceso de experimentación con los prompts.
Implementación y Calidad de la Aplicación de Prueba (15%)

El dashboard en Streamlit/Gradio es funcional, está bien presentado y muestra claramente las tres funcionalidades.
Código limpio y bien organizado en el repositorio de GitHub.
Documentación Escrita (10%)

Informe claro, bien estructurado y que refleje profundidad en el análisis técnico.
Presentación Oral y Demostración (10%)

Claridad en la exposición de la arquitectura y decisiones técnicas.
Demostración en vivo fluida y convincente.