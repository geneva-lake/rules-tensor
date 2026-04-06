# Rule Evaluation with CubeCL (WGPU)

Generalization of case from my practice. We were making reward based loyalty program and we needed to make a system for processing our partners' reports and awarding bonuses to users. With different partners there were different afreements about points accrual. With one, for example, we should award points equal to 5% of the order amount if amount was more than 10 dollars. We created "rules", a mini programming language to process reports flexibly according to contracts. Rule consists of a conditions that checks order compliance and an action that calculates the amount of bonus points.

Here I built a computation pipeline via **CubeCL (WGPU)**, where rules compilated into **CubeCL** kernel at the compile time by proc macro and data is processed as tensors.
At the end application prints bonuses per user.

## Rule Format

`rule_app/src/rule.json` is a single rule object:

```json
{
  "if": { "gte": { "value": 10 } },
  "then": { "multiply": { "value": 0.05 } }
}
```

Supported operators:
- `gte`
- `lte`

Supported actions:
- `multiply`
- `plus`

## Input Data

`rule_app/src/purchase_items.json` contains an array of orders. Each order has fields like `price`, `quantity`, and `user_id`.

## Run

```bash
cargo run -p rule_app
```

The output prints:

```
user_id: <uuid> bonuses: <value>
```

## Notes

- This build is **WGPU-only**. A compatible GPU backend (Metal on macOS) is required.
- The rule kernel is generated at compile time from `rule.json` using a proc-macro.
