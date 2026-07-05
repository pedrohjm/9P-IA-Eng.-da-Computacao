# IA — Trabalhos

Engenharia da Computação — IFTM
Disciplina de Inteligência Artificial — Prof. Jefferson

Coleção de trabalhos práticos da disciplina, cobrindo Redes Neurais Artificiais e Algoritmos Genéticos. Cada trabalho é uma aplicação web independente (HTML + JavaScript puro, sem build), reunida no [hub-trabalhos.html](hub-trabalhos.html) — um hub com barra lateral que carrega cada trabalho num iframe (exceto o Trabalho 06, que abre em aba nova por exigir seleção de pasta local).

## Estrutura do projeto

```
Trabalho NN/        → HTML + JS de cada trabalho
assets/docs/        → slides e enunciados em PDF de cada aula/trabalho
assets/datasets/    → datasets usados nos treinamentos (CSV, imagens)
assets/scripts/     → scripts auxiliares de preparo de dados
hub-trabalhos.html  → hub de navegação entre os trabalhos
```

## Redes Neurais Artificiais

### Trabalho 01 — Regra de Hebb
Aprendizado de portas lógicas bipolares (AND, OR, NAND, NOR) usando a regra de Hebb, a forma mais simples de ajuste de pesos em uma rede neural.

### Trabalho 02 — Perceptron Simples
Reconhecimento das letras A e B representadas em matrizes bipolares 7×7, treinado com o algoritmo do Perceptron simples (uma camada, sem função de ativação não-linear complexa).

### Trabalho 03 — MADALINE
Extensão do Perceptron para reconhecimento de 13 letras (A–M) em matrizes bipolares 7×7, usando múltiplos Adalines combinados (Multiple ADAptive LINear Elements).

### Trabalho 04 — MLP Backpropagation
Multilayer Perceptron com retropropagação de erro (ativação tanh) e topologia configurável, permitindo experimentar diferentes arquiteturas de rede.

### Trabalho 05 — MLP Reconhecimento de Letras
Demo em JavaScript com TensorFlow.js: o usuário desenha uma letra à mão livre ou treina a rede a partir de um CSV (label + 784 pixels, imagens 28×28).

### Trabalho 06 — CNN de Letras
Rede Neural Convolucional (convolução + pooling + camada densa) implementada manualmente em JavaScript, treinada com o dataset `by_class` (NIST, 62 classes: dígitos e letras maiúsculas/minúsculas). Também permite desenhar uma letra e testar a predição.

### Trabalho 07 — MLP Previsão de Cotações (VIVA3)
Multilayer Perceptron com uma camada oculta (ativação tanh) usado para prever o próximo valor de fechamento de uma ação a partir do fechamento anterior — uma aplicação de série temporal simples.

## Algoritmos Genéticos

### Trabalho 08 — Função Objetivo
Introdução aos AGs: minimização de uma função matemática contínua via seleção natural, cruzamento e mutação.

### Trabalho 09 — AG Parametrizável
AG configurável com seleção por torneio, elitismo e cruzamento em dois pontos, otimizando f(x) = −|x · sin(√|x|)| no intervalo [0, 512].

### Trabalho 10 — Horário Escolar
Uso de AG para montar uma grade de horários escolares, representando a solução como um cromossomo matricial (matriz de alocação de aulas).

### Trabalho 11 — Caixeiro Viajante
Problema do Caixeiro Viajante (TSP) aplicado a cidades do Triângulo Mineiro, resolvido com seleção por roleta.

### Trabalho 13 — Rastrigin Real
AG com cromossomos de valores reais, testando diferentes operadores de cruzamento (aritmético, BLX-α, SBX) e mutação (gaussiana, uniforme) para minimizar a função de Rastrigin.

### Trabalho 14 — Ranking TSP
Nova abordagem ao Caixeiro Viajante, agora com seleção por ranking (linear e exponencial), cruzamento OX (Order Crossover) e mutação por troca.

## Como usar

Abra o [hub-trabalhos.html](hub-trabalhos.html) no navegador e escolha um trabalho na barra lateral. Cada trabalho também pode ser aberto isoladamente, direto do seu arquivo HTML dentro da respectiva pasta `Trabalho NN/`.
