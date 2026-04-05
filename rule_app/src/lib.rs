use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use rule_macros::include_rule_kernel;
use serde::Deserialize;

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
    output: &mut Array<F>
) {
    unreachable!()
}

pub fn load_orders_from_file(path: &str) -> Vec<Order> {
    let data = std::fs::read_to_string(path).expect("failed to read purchase_items.json");
    serde_json::from_str(&data).expect("purchase_items.json invalid")
}

pub fn prices_from_orders(orders: &[Order]) -> Vec<f32> {
    orders.iter().map(|o| o.price).collect()
}

pub fn run_rule_on_gpu(prices: &[f32]) -> Vec<f32> {
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

    let cube_dim = CubeDim::new_1d(prices.len() as u32);
    let cube_count = calculate_cube_count_elemwise(&client, prices.len(), cube_dim);

    unsafe {
        apply_rule_kernel::launch_unchecked::<f32, WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(&input_handle, prices.len(), vectorization),
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
    run_rule_on_gpu(&prices)
}

pub fn run_with_orders(path: &str) -> Vec<(String, f32)> {
    let orders = load_orders_from_file(path);
    let prices = prices_from_orders(&orders);
    let bonuses = run_rule_on_gpu(&prices);

    orders
        .into_iter()
        .zip(bonuses.into_iter())
        .map(|(order, bonus)| (order.user_id, bonus))
        .collect()
}
