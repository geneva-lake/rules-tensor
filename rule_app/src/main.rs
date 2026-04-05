fn main() {
    let results = rule_app::run_with_orders(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/purchase_items.json"
    ));
    for (user_id, bonus) in results {
        println!("user_id: {} bonuses: {}", user_id, bonus);
    }
}
