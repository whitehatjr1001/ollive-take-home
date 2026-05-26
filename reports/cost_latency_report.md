# Cost and Latency Report

Latency and throughput are measured from the benchmark run and may vary by cold starts, provider load, network, prompt length, and concurrency.

| Provider | Model | Scenario | Concurrency | Requests | p50 latency | p95 latency | Output tok/s | Cost/request | Cost/1M tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OSS Modal | oss-assistant | warm | 1 | 5 | 2756 ms | 3481 ms | 53.2 | $0.000662 | $1.57 |
| OSS Modal | oss-assistant | batch | 4 | 16 | 2884 ms | 3487 ms | 209.7 | $0.000175 | $0.41 |
| Frontier OpenAI | gpt-4.1 | warm | 1 | 5 | 3370 ms | 5275 ms | 20.6 | $0.001129 | $3.47 |
| Frontier OpenAI | gpt-4.1 | batch | 4 | 16 | 3602 ms | 4327 ms | 82.3 | $0.001117 | $3.45 |