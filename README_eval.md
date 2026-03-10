# PEC Ablation Evaluation

`evaluate_pec.py`는 학습과 동일한 4:4:2 blended dataset 규칙과 seed를 사용해 **학습에 쓰이지 않은 held-out eval split**을 재구성하고, 아래 실험군을 한 번에 평가합니다.

## 실험군

기본 비교축은 **3개 시나리오 × thinking OFF/ON** 입니다.

1. **Base Qwen (No Hint)**
   - 입력: 원문 전체 + 질문
   - 기본 비교 모델: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`
2. **PEC (With Hint)**
   - 입력 순서: **압축된 64 soft prompts 먼저** + 원문 전체 + 질문
3. **PEC (Masked + Hint)**
   - 입력 순서: **압축된 64 soft prompts 먼저** + 핵심 정보 30% 마스킹된 원문 + 질문
   - soft prompt는 항상 **원본 전체 prompt**에서 압축됩니다.

각 시나리오는 Qwen chat template의 `enable_thinking=False/True`를 모두 평가합니다.

## 출력

실행 결과는 `outputs/pec_eval_ablation/<timestamp>/` 아래에 저장됩니다.

- `run_config.json`: 실행 설정
- `summary.csv`, `summary.json`: 실험군별 평균 점수
- `base_no_hint__thinking_off__*.jsonl`
- `base_no_hint__thinking_on__*.jsonl`
- `pec_with_hint__thinking_off__*.jsonl`
- `pec_with_hint__thinking_on__*.jsonl`
- `pec_masked_hint__thinking_off__*.jsonl`
- `pec_masked_hint__thinking_on__*.jsonl`

샘플별 JSONL에는 예측, 정답, `exact_match`, `token_f1`, `rougeL_f1`, 그리고 PEC의 경우 gate 통계가 함께 저장됩니다.
또한 `raw_prediction`에는 원본 생성 결과를, `prediction`에는 선택적으로 `<think>...</think>` 구간을 제거한 채점용 답변을 저장합니다.

## 기본 실행

```bash
python evaluate_pec.py
```

## 빠른 스모크 테스트

```bash
python evaluate_pec.py --max-samples 8 --max-new-tokens 64
```

## thinking ON만 따로 보기

```bash
python evaluate_pec.py --thinking-modes on --max-samples 200
```

## Base 모델 일부만 선택

```bash
python evaluate_pec.py \
  --base-models Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
  --max-samples 200
```

## PEC만 다시 평가

```bash
python evaluate_pec.py --skip-base --max-samples 200
```

## held-out eval JSONL도 함께 저장

```bash
python evaluate_pec.py \
  --persist-heldout-jsonl data/processed_pec_dataset/heldout_eval.jsonl
```

## 메모

- 기본값은 학습 코드와 같은 `seed=42`, `epoch_size=200000`, `eval_ratio=0.02`를 사용합니다.
- `PEC (Masked + Hint)`의 마스킹은 `EntityMasker(mask_prob=0.3)`를 고정 seed로 적용해 재현 가능하게 만들었습니다.
- 큰 Base 모델(`4B`, `8B`)은 메모리 사용량이 크므로 필요하면 `--base-models`로 부분 실행하세요.
- macOS/MPS 환경에서는 대형 모델이 느리거나 메모리가 부족할 수 있습니다.
- 학습 코드(`train_pec.py`)에서는 `composer_enable_thinking=False`로 학습됐지만, 이 평가는 **추론 시점**에 thinking ON/OFF를 모두 비교합니다.
