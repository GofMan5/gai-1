# Eval Gates

Каждое дообучение GAI-1 должно проходить gates:

- русский чат и стиль;
- instruction following;
- reasoning/math;
- code patch eval;
- tool-use;
- hallucination pressure;
- safety regression;
- latency/cost.

Цель: модель можно дообучать в любой момент, но нельзя продвигать checkpoint без сравнения с предыдущей версией.

