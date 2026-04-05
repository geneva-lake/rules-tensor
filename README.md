# Rule Tensor + Tucker Decomposition (WGPU)

Generalization of case from my practice. We were making reward based loyalty program and we needed to make a system for processing our partners' reports and awarding bonuses to users.
With different partners there were different afreements about points accrual. With one, for example, we should award points equal to 5% of the order amount if amount was more than 10 dollars.
We created "rules", a mini programming language to process reports flexibly according to contracts. Rule consists of a conditions that checks order compliance and an action that calculates the amount of bonus points.

Here I built a **rule tensor** from purchase data, applied **Tucker decomposition** to learn latent structure over rule axes, and used those factors to interpret category groupings. The computation pipeline is GPU-accelerated via **CubeCL (WGPU)**.

## What Is the Rule Tensor?

We represent rules as a 3D tensor with axes:

- `condition_type` (e.g., `price_gte`, `quantity_lte`, `quantity_gte`)
- `action_type` (e.g., `multiply`, `plus`)
- `category` (e.g., `Electronics`, `Hygiene`, `Books`)

Each tensor entry stores a **weight** that reflects how often a rule is triggered within a category.

### Weight Definition

For each rule and category, the weight is:

`weight = passed / total`

Where:

- `total` = number of orders in that category
- `passed` = number of orders where the rule condition is true

If multiple rules map to the same `(condition_type, action_type, category)` cell, their weights are averaged.

## Tucker Decomposition

Tucker decomposition factorizes the tensor into:

- a **core tensor** `G`
- factor matrices for each axis: `U_condition`, `U_action`, `U_category`

This provides two practical benefits:

- **Compression**: store a smaller core and factor matrices instead of the full tensor
- **Interpretability**: inspect factor loadings to discover which categories group together

## Category Grouping and Importance

We interpret the **category factor matrix** to find latent groups and measure importance:

- **Groups**: each category is assigned to the component where its absolute loading is maximal
- **Importance**: for each category, we sum the absolute loadings across all components

The CLI prints:

- Category groups from Tucker factors
- Top categories by Tucker importance

## Inputs

- `rule_app/src/rule.json`
  - Array of rules applied in sequence
  - Supports fields: `price`, `quantity`
  - Supports operators: `gte`, `lte`

- `rule_app/src/purchase_items.json`
  - Orders used to build the rule tensor

## Run

```bash
cargo run -p rule_app
```

### Output

The program prints only:

- Tucker category groups
- Top categories by Tucker importance

## Notes

- This build is **WGPU-only**. A compatible GPU backend (Metal on macOS) is required.
- The rule kernel is generated at compile time from `rule.json` using a proc-macro.

