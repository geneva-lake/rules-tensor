fn main() {
    let orders = rule_app::load_orders_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/purchase_items.json"
    ));
    let rules = rule_app::load_rule_spec_from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/rule.json"
    ));
    let tensor = rule_app::build_rule_tensor(&orders, &rules);

    let ranks = rule_app::default_tucker_ranks(&tensor);

    let groups = rule_app::tucker_category_groups(&tensor, &ranks);
    println!("\nCategory groups from Tucker factors:");
    for group in groups {
        println!("component {}", group.component);
        for (cat, weight) in group.categories {
            println!("  {} (loading {:.4})", cat, weight);
        }
    }

    let importance = rule_app::tucker_category_importance(&tensor, &ranks);
    println!("\nTop categories by Tucker importance:");
    for (cat, score) in importance.iter().take(5) {
        println!("  {} (score {:.4})", cat, score);
    }
}
