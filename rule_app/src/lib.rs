use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use rule_macros::include_rule_kernel;
use serde::Deserialize;
use std::collections::BTreeMap;
use tenrso_core::DenseND;
use tenrso_decomp::tucker_hosvd;

#[derive(Debug, Clone, Copy)]
pub struct Gte {
    pub value: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Lte {
    pub value: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Condition {
    Gte(Gte),
    Lte(Lte),
}

impl Condition {
    pub fn to_params(self) -> (u32, f32) {
        match self {
            Condition::Gte(x) => (0, x.value),
            Condition::Lte(x) => (1, x.value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Plus {
    pub value: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Multiply {
    pub value: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum Action {
    Plus(Plus),
    Multiply(Multiply),
}

impl Action {
    pub fn to_params(self) -> (u32, f32) {
        match self {
            Action::Plus(x) => (0, x.value),
            Action::Multiply(x) => (1, x.value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rule {
    pub if_: Condition,
    pub then: Action,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RuleSpec {
    #[serde(rename = "if")]
    if_: RuleIf,
    then: RuleAction,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RuleIf {
    pub field: String,
    #[serde(default)]
    pub gte: Option<RuleValue>,
    #[serde(default)]
    pub lte: Option<RuleValue>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RuleValue {
    pub value: f32,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum RuleAction {
    Plus { value: f32 },
    Multiply { value: f32 },
}

#[derive(Debug, Deserialize)]
pub struct Order {
    pub id: u32,
    pub user_id: String,
    pub category: String,
    pub name: String,
    pub brand: String,
    pub quantity: i32,
    pub price: f32,
    pub currency: String,
}

#[include_rule_kernel("src/rule.json")]
#[cube(launch_unchecked)]
fn apply_rule_kernel<F: Float>(
    prices: &Array<F>,
    quantities: &Array<F>,
    output: &mut Array<F>
) {
    unreachable!()
}

pub fn load_orders_from_file(path: &str) -> Vec<Order> {
    let data = std::fs::read_to_string(path).expect("failed to read purchase_items.json");
    serde_json::from_str(&data).expect("purchase_items.json invalid")
}

pub fn load_rule_spec_from_file(path: &str) -> Vec<RuleSpec> {
    let data = std::fs::read_to_string(path).expect("failed to read rule.json");
    serde_json::from_str(&data).expect("rule.json invalid")
}

pub fn prices_from_orders(orders: &[Order]) -> Vec<f32> {
    orders.iter().map(|o| o.price).collect()
}

pub fn quantities_from_orders(orders: &[Order]) -> Vec<f32> {
    orders.iter().map(|o| o.quantity as f32).collect()
}

pub fn run_rule_on_gpu(prices: &[f32], quantities: &[f32]) -> Vec<f32> {
    let device = WgpuDevice::DefaultDevice;
    let client = match std::panic::catch_unwind(|| WgpuRuntime::client(&device)) {
        Ok(client) => client,
        Err(_) => {
            panic!(
                "WGPU adapter not found. This build is WGPU-only; run on a machine with a compatible GPU/Metal backend."
            );
        }
    };

    let vectorization: usize = 1;
    let output_handle = client.empty(prices.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(prices));
    let qty_handle = client.create_from_slice(f32::as_bytes(quantities));

    let cube_dim = CubeDim::new_1d(prices.len() as u32);
    let cube_count = calculate_cube_count_elemwise(&client, prices.len(), cube_dim);

    unsafe {
        apply_rule_kernel::launch_unchecked::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, prices.len(), vectorization),
            ArrayArg::from_raw_parts::<f32>(&qty_handle, quantities.len(), vectorization),
            ArrayArg::from_raw_parts::<f32>(&output_handle, prices.len(), vectorization),
        )
        .unwrap();
    };

    let bytes = client.read_one(output_handle);
    f32::from_bytes(&bytes).to_vec()
}

pub fn run() -> Vec<f32> {
    let orders = load_orders_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/purchase_items.json"
    ));
    let prices = prices_from_orders(&orders);
    let quantities = quantities_from_orders(&orders);
    run_rule_on_gpu(&prices, &quantities)
}

pub fn run_with_orders(path: &str) -> Vec<(String, f32)> {
    let orders = load_orders_from_file(path);
    let prices = prices_from_orders(&orders);
    let quantities = quantities_from_orders(&orders);
    let bonuses = run_rule_on_gpu(&prices, &quantities);

    orders
        .into_iter()
        .zip(bonuses.into_iter())
        .map(|(order, bonus)| (order.user_id, bonus))
        .collect()
}

#[derive(Debug)]
pub struct RuleTensor {
    pub condition_types: Vec<String>,
    pub action_types: Vec<&'static str>,
    pub categories: Vec<String>,
    pub data: Vec<f32>,
}

impl RuleTensor {
    pub fn index(&self, cond: usize, action: usize, category: usize) -> usize {
        let a_len = self.action_types.len();
        let c_len = self.categories.len();
        (cond * a_len + action) * c_len + category
    }
}

pub fn build_rule_tensor(orders: &[Order], rules: &[RuleSpec]) -> RuleTensor {
    let mut condition_types: Vec<String> = Vec::new();
    let action_types: Vec<&'static str> = vec!["plus", "multiply"];

    for rule in rules {
        let op = match (&rule.if_.gte, &rule.if_.lte) {
            (Some(_), None) => "gte",
            (None, Some(_)) => "lte",
            _ => "gte",
        };
        let key = format!("{}_{}", rule.if_.field.to_lowercase(), op);
        if !condition_types.contains(&key) {
            condition_types.push(key);
        }
    }

    let mut category_counts: BTreeMap<String, u32> = BTreeMap::new();
    for order in orders {
        *category_counts.entry(order.category.clone()).or_insert(0) += 1;
    }

    let categories: Vec<String> = category_counts.keys().cloned().collect();
    let mut data = vec![0.0f32; condition_types.len() * action_types.len() * categories.len()];
    let mut counts = vec![0u32; data.len()];

    for rule in rules {
        let op = match (&rule.if_.gte, &rule.if_.lte) {
            (Some(_), None) => "gte",
            (None, Some(_)) => "lte",
            _ => "gte",
        };
        let cond_key = format!("{}_{}", rule.if_.field.to_lowercase(), op);
        let cond_idx = condition_types
            .iter()
            .position(|c| c == &cond_key)
            .unwrap();
        let action_idx = match rule.then {
            RuleAction::Plus { .. } => 0,
            RuleAction::Multiply { .. } => 1,
        };

        let mut passed_by_category: BTreeMap<String, u32> = BTreeMap::new();
        for order in orders {
            let value = match rule.if_.field.as_str() {
                "price" => order.price,
                "quantity" => order.quantity as f32,
                _ => order.price,
            };
            let passed = match (&rule.if_.gte, &rule.if_.lte) {
                (Some(v), None) => value > v.value,
                (None, Some(v)) => value < v.value,
                _ => false,
            };
            if passed {
                *passed_by_category.entry(order.category.clone()).or_insert(0) += 1;
            }
        }

        for (cat_idx, cat) in categories.iter().enumerate() {
            let total = *category_counts.get(cat).unwrap_or(&0);
            let passed = *passed_by_category.get(cat).unwrap_or(&0);
            let weight = if total == 0 { 0.0 } else { passed as f32 / total as f32 };
            let idx = (cond_idx * action_types.len() + action_idx) * categories.len() + cat_idx;
            data[idx] += weight;
            counts[idx] += 1;
        }
    }

    for i in 0..data.len() {
        if counts[i] > 0 {
            data[i] /= counts[i] as f32;
        }
    }

    RuleTensor {
        condition_types,
        action_types,
        categories,
        data,
    }
}

pub struct TuckerSummary {
    pub core_shape: Vec<usize>,
    pub factor_shapes: Vec<(usize, usize)>,
    pub compression_ratio: f64,
}

pub fn tucker_decompose_rule_tensor(tensor: &RuleTensor, ranks: &[usize]) -> TuckerSummary {
    let shape = [
        tensor.condition_types.len(),
        tensor.action_types.len(),
        tensor.categories.len(),
    ];
    let data_f64: Vec<f64> = tensor.data.iter().map(|&v| v as f64).collect();
    let dense = DenseND::<f64>::from_vec(data_f64, &shape).expect("tensor reshape failed");

    let tucker = tucker_hosvd(&dense, ranks).expect("tucker_hosvd failed");
    let core_shape = tucker.core.shape().to_vec();
    let factor_shapes = tucker
        .factors
        .iter()
        .map(|f| {
            let s = f.shape();
            (s[0], s[1])
        })
        .collect();
    let compression_ratio = tucker.compression_ratio();

    TuckerSummary {
        core_shape,
        factor_shapes,
        compression_ratio,
    }
}

pub fn default_tucker_ranks(tensor: &RuleTensor) -> Vec<usize> {
    let c = tensor.condition_types.len().min(2);
    let a = tensor.action_types.len().min(2);
    let k = tensor.categories.len().min(3);
    vec![c, a, k]
}

pub fn tucker_reconstruct_rule_tensor(tensor: &RuleTensor, ranks: &[usize]) -> RuleTensor {
    let shape = [
        tensor.condition_types.len(),
        tensor.action_types.len(),
        tensor.categories.len(),
    ];
    let data_f64: Vec<f64> = tensor.data.iter().map(|&v| v as f64).collect();
    let dense = DenseND::<f64>::from_vec(data_f64, &shape).expect("tensor reshape failed");

    let tucker = tucker_hosvd(&dense, ranks).expect("tucker_hosvd failed");
    let reconstructed = tucker.reconstruct().expect("tucker reconstruct failed");
    let recon_data: Vec<f32> = reconstructed.to_vec().into_iter().map(|v| v as f32).collect();

    RuleTensor {
        condition_types: tensor.condition_types.clone(),
        action_types: tensor.action_types.clone(),
        categories: tensor.categories.clone(),
        data: recon_data,
    }
}

pub fn apply_tucker_weights_to_bonuses(
    orders: &[Order],
    bonuses: &[f32],
    rules: &[RuleSpec],
    approx_tensor: &RuleTensor,
) -> Vec<f32> {
    orders
        .iter()
        .zip(bonuses.iter())
        .map(|(order, bonus)| {
            let cat_idx = approx_tensor
                .categories
                .iter()
                .position(|c| c == &order.category)
                .unwrap_or(0);

            let mut weight = 1.0f32;
            for rule in rules {
                let op = match (&rule.if_.gte, &rule.if_.lte) {
                    (Some(_), None) => "gte",
                    (None, Some(_)) => "lte",
                    _ => "gte",
                };
                let cond_key = format!("{}_{}", rule.if_.field.to_lowercase(), op);
                let cond_idx = approx_tensor
                    .condition_types
                    .iter()
                    .position(|c| c == &cond_key)
                    .unwrap_or(0);
                let action_idx = match rule.then {
                    RuleAction::Plus { .. } => 0,
                    RuleAction::Multiply { .. } => 1,
                };
                let idx = approx_tensor.index(cond_idx, action_idx, cat_idx);
                weight *= approx_tensor.data[idx].max(0.0);
            }

            bonus * weight
        })
        .collect()
}

pub struct CategoryGroup {
    pub component: usize,
    pub categories: Vec<(String, f64)>,
}

pub fn tucker_category_groups(tensor: &RuleTensor, ranks: &[usize]) -> Vec<CategoryGroup> {
    let shape = [
        tensor.condition_types.len(),
        tensor.action_types.len(),
        tensor.categories.len(),
    ];
    let data_f64: Vec<f64> = tensor.data.iter().map(|&v| v as f64).collect();
    let dense = DenseND::<f64>::from_vec(data_f64, &shape).expect("tensor reshape failed");

    let tucker = tucker_hosvd(&dense, ranks).expect("tucker_hosvd failed");
    let cat_factors = tucker
        .factors
        .get(2)
        .expect("missing category factor");

    let cat_shape = cat_factors.shape();
    let cat_count = cat_shape[0];
    let comp_count = cat_shape[1];
    let cat_data = cat_factors.iter().cloned().collect::<Vec<f64>>();

    let mut groups: Vec<CategoryGroup> = (0..comp_count)
        .map(|c| CategoryGroup {
            component: c,
            categories: Vec::new(),
        })
        .collect();

    for i in 0..cat_count {
        let mut best_comp = 0usize;
        let mut best_val = cat_data[i * comp_count].abs();
        for j in 1..comp_count {
            let v = cat_data[i * comp_count + j].abs();
            if v > best_val {
                best_val = v;
                best_comp = j;
            }
        }
        let signed = cat_data[i * comp_count + best_comp];
        groups[best_comp]
            .categories
            .push((tensor.categories[i].clone(), signed));
    }

    for g in groups.iter_mut() {
        g.categories
            .sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    }

    groups
}

pub fn tucker_category_importance(tensor: &RuleTensor, ranks: &[usize]) -> Vec<(String, f64)> {
    let shape = [
        tensor.condition_types.len(),
        tensor.action_types.len(),
        tensor.categories.len(),
    ];
    let data_f64: Vec<f64> = tensor.data.iter().map(|&v| v as f64).collect();
    let dense = DenseND::<f64>::from_vec(data_f64, &shape).expect("tensor reshape failed");

    let tucker = tucker_hosvd(&dense, ranks).expect("tucker_hosvd failed");
    let cat_factors = tucker
        .factors
        .get(2)
        .expect("missing category factor");

    let cat_shape = cat_factors.shape();
    let cat_count = cat_shape[0];
    let comp_count = cat_shape[1];
    let cat_data = cat_factors.iter().cloned().collect::<Vec<f64>>();

    let mut scores = Vec::with_capacity(cat_count);
    for i in 0..cat_count {
        let mut sum = 0.0f64;
        for j in 0..comp_count {
            sum += cat_data[i * comp_count + j].abs();
        }
        scores.push((tensor.categories[i].clone(), sum));
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores
}

pub fn format_rule_tensor_table(tensor: &RuleTensor) -> String {
    let mut lines = Vec::new();
    let mut headers = vec!["condition".to_string(), "action".to_string()];
    headers.extend(tensor.categories.iter().cloned());
    lines.push(headers.join("\t"));

    for (cond_idx, cond) in tensor.condition_types.iter().enumerate() {
        for (act_idx, act) in tensor.action_types.iter().enumerate() {
            let mut row = vec![cond.to_string(), act.to_string()];
            for (cat_idx, _cat) in tensor.categories.iter().enumerate() {
                let idx = tensor.index(cond_idx, act_idx, cat_idx);
                row.push(format!("{:.4}", tensor.data[idx]));
            }
            lines.push(row.join("\t"));
        }
    }

    lines.join("\n")
}
