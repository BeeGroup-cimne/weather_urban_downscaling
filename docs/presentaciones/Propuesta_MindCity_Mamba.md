# Propuesta de Presentación: Climate-Ready BCN & Vision Mamba

---

## Diapositiva 1: Título de la Propuesta

**Título:** Super-resolución eficiente de microclimas urbanos para la adaptación al calor.

**Subtítulo:** Integración de la plataforma MindCity con modelos State-Space (Vision Mamba) hiperlocales.

**Texto sugerido (Guion para el ponente):** 
"Hoy presentaremos cómo estamos superando las limitaciones actuales en la predicción del clima urbano. Unificando la robusta infraestructura de datos de Barcelona con arquitecturas de Inteligencia Artificial de próxima generación, podemos predecir el impacto de las olas de calor calle a calle, en tiempo real."

**Gráfico sugerido:** 
Incluir la imagen generada en el modelo:
`experiments/heatwaves/publish_run_20260220_220458/figures/tiles_publish_PUB_MAMBA_S42_2017-08-15_15_00_00.png`

---

## Diapositiva 2: Base: Climate-Ready BCN

**Título:** La Fundación de Datos: Climate-Ready BCN y MindCity

**Texto sugerido (Guion):** 
"Para modelar la ciudad, primero necesitamos entenderla. La plataforma MindCity en Barcelona nos provee un ecosistema real de datos. Utilizamos 61,000 registros de edificios residenciales, combinando catastro, eficiencia energética y redes de sensores. El gran reto era armonizar datos tan heterogéneos, lo cual logramos mediante Grafos de Conocimiento (KG) y la estructura de la Ontología BIGG."

**Gráfico sugerido:** 
Extraer de `mindcity_eng_lite.pdf` el **diagrama de la Ontología BIGG** o el esquema del **Grafo de Conocimiento**.

---

## Diapositiva 3: El Desafío de la Resolución

**Título:** El límite de los 9 Kilómetros

**Texto sugerido (Guion):** 
"Aquí radica el problema principal. Los sistemas climáticos actuales, como ERA5-Land, nos dan datos a 9.0 km de resolución. Esto no nos sirve para entender la resiliencia urbana. Además, los intentos anteriores de hacer 'downscaling' han fallado: los modelos tradicionales ignoraban la inercia térmica de los edificios (clave en zonas densas), y las arquitecturas recurrentes (ConvLSTM) resultaban insostenibles por su enorme coste computacional."

**Gráfico sugerido:** 
De `mindcity_eng_lite.pptx`, incluir el **modelo RC de demanda térmica** o una comparativa de un píxel de 9km.

---

## Diapositiva 4: Metodología: U-Net Híbrida

**Título:** La Solución: U-Net Híbrida Espaciotemporal

**Texto sugerido (Guion):** 
"Nuestra propuesta es una nueva arquitectura U-Net híbrida. En lugar de procesar solo características atmosféricas, nuestro modelo inyecta variables de morfología urbana de altísima resolución (calles, alturas de edificios, vegetación) junto con los datos meteorológicos de baja resolución. Esto fuerza a la red a predecir la temperatura exacta basándose en cómo el entorno físico local reacciona al clima."

**Gráfico sugerido:** 
Un diagrama estructural de la red **U-Net** detallada en el abstract.

---

## Diapositiva 5: Mamba vs. ConvLSTM

**Título:** Evolución Computacional: Incorporando Vision Mamba

**Texto sugerido (Guion):** 
"La verdadera magia ocurre en el corazón matemático del modelo. Sustituimos las redes recurrentes tradicionales del *bottleneck* por State Space Models (específicamente Vision Mamba). Mientras arquitecturas previas escalan de forma cuadrática y colapsan la memoria, Mamba ofrece **complejidad lineal**. Mantenemos una memoria perfecta de la inercia térmica temporal de los edificios, pero con una fracción diminuta del costo de hardware."

**Gráfico sugerido:** 
Las **gráficas de estabilidad de métricas y tiempos de entrenamiento/memoria** del Experimento 1.

---

## Diapositiva 6: Resultados y Escalabilidad

**Título:** Precisión Hiperlocal Sin Cuellos de Botella

**Texto sugerido (Guion):** 
"Los resultados avalan el modelo: la arquitectura SSM es drásticamente más eficiente y su precisión es superior en episodios críticos como las olas de calor de agosto. Al romper el cuello de botella del costo computacional, esta arquitectura es escalable a toda el Área Metropolitana. No es solo un modelo de alta resolución, es un modelo capaz de operar de manera continua, habilitando un escaneo constante de la ciudad."

**Gráfico sugerido:** 
Una comparativa de los resultados del Experimento 1 (15 de agosto a las 15:00 hrs):
`PUB_BASELINE_BILINEAR_2017-08-15_15_00_00.png` vs. `PUB_MAMBA_S42_2017-08-15_15_00_00.png`.

---

## Diapositiva 7: Integración Futura

**Título:** Del Laboratorio al Ciudadano

**Texto sugerido (Guion):** 
"¿Para qué sirve un microclima en tiempo real? La eficiencia en inferencia de la red U-Net Mamba nos permite integrarla de vuelta al ecosistema MindCity como un servicio vivo. El objetivo a corto plazo es conectar estos mapas de estrés térmico calle a calle directamente a los sistemas ciudadanos, alimentando las alertas tempranas en plataformas reales como la App *'La Meva Energia'*."

**Gráfico sugerido:** 
Un mockup de **La App "La Meva Energia"** (extraído de `mindcity_eng_lite.pdf`).
