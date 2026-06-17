---
marp: true
theme: default
paginate: true
size: 16:9
header: "Diagnóstico de enfermedades en hoja de papa con hardware de consumo"
footer: "Niels Pacheco · MIA-07 Aprendizaje Profundo · 2026"
style: |
  section { font-size: 23px; justify-content: flex-start; padding: 48px 60px 58px 60px; }
  h1 { color: #1b5e20; font-size: 40px; }
  h2 { color: #2e7d32; font-size: 30px; }
  h3 { font-size: 24px; }
  table { font-size: 21px; margin: 8px auto; }
  img { display: block; margin: 8px auto; }
  ul, ol { margin-top: 6px; }
  li { margin-bottom: 5px; }
  .small { font-size: 18px; }
  section.lead { justify-content: center; text-align: center; }
  section.lead h1 { font-size: 44px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _header: "" -->
<!-- _footer: "" -->

# 🥔🔬 CNN ligeras para el diagnóstico de enfermedades en hoja de papa con hardware de consumo

### Entrenamiento desde cero vs. aprendizaje por transferencia, con interpretabilidad Grad-CAM

**Niels Pacheco**
MIA-07 — Redes Neuronales y Aprendizaje Profundo (Sección C)
Proyecto Final · Junio de 2026

---

## ¿Por qué la papa? ¿Por qué ahora?

- La **papa** (*Solanum tuberosum*) se domesticó en los **Andes peruanos** hace ~7,000 años y es el **3.er cultivo alimentario** más importante del mundo.
- El **tizón tardío** (*Phytophthora infestans*) causó la hambruna irlandesa y puede **arrasar un campo en días**; el **tizón temprano** (*Alternaria solani*) también está muy extendido.
- El diagnóstico experto es **escaso en las comunidades rurales andinas** — justo donde se concentra el cultivo.
- El aprendizaje profundo podría **democratizar el diagnóstico desde una sola foto de hoja** — *si* corre en el hardware que los agricultores realmente tienen.

---

## La pregunta de investigación

> **¿Qué tan cerca puede llegar una CNN pequeña entrenada _desde cero_ en una laptop sin GPU al aprendizaje por transferencia preentrenado en ImageNet** para clasificar enfermedades en hoja de papa?

Dos vacíos en la literatura que abordamos:

1. La mayoría de los sistemas asume **GPU + grandes redes preentrenadas**.
2. Las exactitudes destacadas rara vez vienen con **intervalos de confianza, calibración, o una verificación de que el modelo mire la enfermedad**.

---

## Contribuciones

1. Una **CNN propia compacta** (0.33 M parámetros) entrenable de extremo a extremo en CPU/Apple Silicon, comparada de frente con transferencia (MobileNetV2, ResNet-18, EfficientNet-B0 congeladas).
2. Una **evaluación estadísticamente rigurosa**: múltiples semillas, **IC bootstrap al 95%** y **prueba pareada de McNemar** — no un único número.
3. Un **análisis de calibración** (ECE + diagramas de fiabilidad), rara vez reportado en este dominio.
4. Un estudio de **interpretabilidad Grad-CAM** que sondea el **sesgo de fondo** de PlantVillage.
5. Una **publicación de código abierto totalmente reproducible** (dependencias fijadas, semillas fijas, un comando).

---

## Datos: PlantVillage (subconjunto de papa)

- **2,152 imágenes**, 3 clases: **sana / tizón temprano / tizón tardío**
- Fotos controladas en laboratorio, **clases desbalanceadas** (pocas sanas)
- **Partición estratificada fija** 70/15/15 → **1506 / 323 / 323** imágenes
- *El mismo conjunto de prueba para cada modelo* — requisito para una prueba de significancia válida
- ⚠️ Los fondos son muy uniformes — volveremos a esto al final

---

## Datos: ejemplos por clase

![h:330](../results/figures/dataset_samples.png)

<span class="small">Imágenes representativas del conjunto de prueba reservado. Note los fondos uniformes: una propiedad que, como mostraremos, los modelos pueden explotar.</span>

---

## Cuatro modelos, una comparación justa

| Modelo | Origen | Parámetros |
|-------|--------|--------|
| **CNN propia** | **desde cero** | ~0.33 M |
| MobileNetV2 | ImageNet, congelada + cabezal nuevo | ~2.2 M |
| ResNet-18 | ImageNet, congelada + cabezal nuevo | ~11 M |
| EfficientNet-B0 | ImageNet, congelada + cabezal nuevo | ~4 M |

- Mismo preprocesamiento y normalización ImageNet para todos → **sólo cambian arquitectura/pesos**
- En transferencia: **red base congelada**, se entrena sólo un cabezal de clasificación nuevo

---

## La CNN propia (desde cero)

- **4 bloques convolucionales**: cada bloque = 2 × (Conv 3×3 + BatchNorm + ReLU) + max pooling
- Anchos de canal **24 → 48 → 96 → 96** (crecen y se estabilizan para mantenerla compacta)
- **Global average pooling** → dropout 0.3 → clasificador lineal
- **328,587 parámetros totales**, *todos entrenables* de extremo a extremo
- Entradas 128×128, normalización ImageNet — idéntica a las líneas base

---

## Metodología — entrenamiento

- **AdamW** + planificación coseno de la tasa de aprendizaje
- Entropía cruzada **ponderada por clase** + **suavizado de etiquetas** (contra el desbalance)
- **Parada temprana** sobre el F1 macro de validación
- **3 semillas aleatorias** por modelo → reportamos media ± desviación estándar
- Entrenado por completo en **Apple Silicon (MPS), sin GPU**

---

## Metodología — evaluación y estadística

- **Exactitud** y **F1 macro** (robusto al desbalance), media ± DE sobre 3 semillas
- **IC bootstrap** no paramétrico al 95% por corrida (incertidumbre dentro del test)
- **Prueba pareada de McNemar** sobre el test compartido (binomial exacta si hay pocas discordancias)
- **Calibración**: error de calibración esperado (ECE) + diagramas de fiabilidad
- **Grad-CAM** para ver *dónde mira* cada modelo

---

## `/autoresearch`: búsqueda autónoma de hiperparámetros

- **Búsqueda aleatoria** con semilla, 10 ensayos, seleccionando por F1 macro de **validación** (nunca el test)
- Espacio: tasa de aprendizaje, decaimiento de pesos, ancho, dropout, intensidad de aumentación
- La mejor configuración se integró en `configs/custom_cnn.yaml`
- Trayectoria completa registrada → `autoresearch.jsonl` + dashboard + bitácora

<span class="small">Mejor F1 macro de validación: **96.9%** (ensayo 9: lr 1e-3, wd 1e-5, ancho 24, dropout 0.3, aug 0.5)</span>

---

## Resultados — tabla principal

| Modelo | Params | Exactitud (%) | F1 macro (%) | ECE |
|--------|-------:|:---:|:---:|:---:|
| **CNN propia (desde cero)** | **328,587** | **97.3 ± 2.3** | **96.7 ± 2.1** | **0.123** |
| EfficientNet-B0 | 4,011,391 | 94.9 ± 0.5 | 91.7 ± 1.4 | 0.157 |
| MobileNetV2 | 2,227,715 | 94.9 ± 0.8 | 91.6 ± 1.2 | 0.134 |
| ResNet-18 | 11,178,051 | 91.7 ± 1.5 | 86.5 ± 2.1 | 0.171 |

<span class="small">Media ± DE sobre 3 semillas. La CNN propia es la **mejor en exactitud, F1 macro y calibración** — con 7–34× menos parámetros.</span>

---

## Resultados — la CNN desde cero *gana* 🏆

![h:330](../results/figures/model_comparison.png)

- F1 macro **96.7%** > mejor transferencia **91.7%** (EfficientNet-B0)
- **McNemar:** supera significativamente a **los tres** modelos por transferencia (*p* = 0.017, 0.004, 7e-5)
- Las características congeladas de ImageNet **no están adaptadas al dominio**

---

## Curvas de entrenamiento

![h:340](../results/figures/training_curves.png)

- Las redes base preentrenadas **convergen rápido**; la CNN desde cero converge más lento a un nivel similar
- Todos alcanzan una **meseta alta** dentro del presupuesto de entrenamiento

---

## Estructura de confusión

![h:300](../results/figures/confusion_matrices.png)

- Los errores de la CNN propia son sobre todo **enfermedad ↔ enfermedad** (operativamente benignos — ambos requieren acción)
- ⚠️ La transferencia confunde algo de **tizón tardío → sana** (10–12 de 150): falsos negativos peligrosos
- La CNN propia no sólo puntúa más alto, también **falla de forma más segura**

---

## Calibración

![h:300](../results/figures/reliability_diagrams.png)

- Todos los modelos están **moderadamente mal calibrados** (ECE 0.12–0.17)
- La **CNN propia es la mejor calibrada**; la mala calibración viene sobre todo de la confianza media
- Un sistema desplegado se beneficiaría del **escalado por temperatura** antes de confiar en las probabilidades

---

## 🔍 ¿Dónde miran los modelos?

![h:330](../results/figures/gradcam_panel.png)

- Atienden a las lesiones **pero también a bordes de la hoja y al fondo**
- Consistente con el **sesgo de fondo** de PlantVillage (Noyan 2022; Barbedo 2018)
- ⟹ La exactitud de laboratorio es una **cota superior** del desempeño en campo

---

## Discusión

**Lo que sí podemos afirmar**
- Una CNN pequeña entrenada **desde cero en una laptop** *supera significativamente* a la transferencia de red base congelada — con 7–34× menos parámetros.
- Bajo un **presupuesto de cómputo igualmente bajo**, aprender características del dominio de extremo a extremo vence a reutilizar características congeladas de ImageNet.

**Lo que no debemos afirmar**
- Que la alta exactitud en PlantVillage ⟹ listo para el campo. El **sesgo de fondo** lo socava.
- Que la transferencia sea "peor" en general — el **ajuste fino completo** probablemente cerraría la brecha.

---

## Limitaciones y trabajo futuro

**Limitaciones**
- Imágenes de **laboratorio** con fondos limpios; en campo esperamos degradación en todos los modelos.
- Tarea de **3 clases** comparativamente fácil → comprime las diferencias entre arquitecturas.
- Redes base **congeladas** (no ajuste fino completo); calibración moderada; sin despliegue en dispositivo aún.

**Trabajo futuro**
- Evaluación con **imágenes de campo** · eliminación/segmentación de fondo · benchmarking de **latencia y energía en dispositivo**

---

## Conclusiones y reproducibilidad

- La CNN compacta entrenada **desde cero supera significativamente** a la transferencia congelada aquí (presupuesto de bajo cómputo igual); el ajuste fino completo costaría más.
- **Grad-CAM** revela una dependencia compartida del fondo del conjunto — **honestidad** por encima de perseguir el ranking.
- Todo se reproduce con **un solo comando** (entorno `uv` fijado, semillas fijas, figuras y artículo auto-generados):

```bash
make setup && make data && make train-all && make eval && make figures && make paper
```

**Repositorio:** github.com/nielspac177/papa-vision — **¡Gracias!** · nielspacheco1997@gmail.com
