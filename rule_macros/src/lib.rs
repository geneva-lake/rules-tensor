use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitFloat, LitStr};

#[derive(serde::Deserialize)]
struct RuleConfig {
    #[serde(rename = "if")]
    if_: CondConfig,
    then: ActConfig,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum CondConfig {
    Gte { value: f32 },
    Lte { value: f32 },
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum ActConfig {
    Plus { value: f32 },
    Multiply { value: f32 },
}

#[proc_macro_attribute]
pub fn include_rule_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let path_lit = parse_macro_input!(attr as LitStr);
    let path_value = path_lit.value();

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("include_rule_kernel: CARGO_MANIFEST_DIR not set");
    let full_path = std::path::Path::new(&manifest_dir).join(&path_value);

    let json = std::fs::read_to_string(&full_path).unwrap_or_else(|err| {
        panic!(
            "include_rule_kernel: failed to read JSON file at {}: {}",
            full_path.display(),
            err
        )
    });
    let cfg: RuleConfig = serde_json::from_str(&json).unwrap_or_else(|err| {
        panic!(
            "include_rule_kernel: invalid JSON in {}: {}",
            full_path.display(),
            err
        )
    });

    let (cond_op, cond_value) = match cfg.if_ {
        CondConfig::Gte { value } => (quote!(>), value),
        CondConfig::Lte { value } => (quote!(<), value),
    };

    let (action_op, action_value) = match cfg.then {
        ActConfig::Plus { value } => (quote!(+), value),
        ActConfig::Multiply { value } => (quote!(*), value),
    };

    let cond_value_lit =
        LitFloat::new(&format!("{cond_value}f32"), proc_macro2::Span::call_site());
    let action_value_lit =
        LitFloat::new(&format!("{action_value}f32"), proc_macro2::Span::call_site());

    let mut item_fn = parse_macro_input!(item as ItemFn);

    item_fn.block = Box::new(syn::parse_quote!({
        const _: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #path_lit));
        if ABSOLUTE_POS < prices.len() {
            let price = prices[ABSOLUTE_POS];
            let cond_value = F::new(#cond_value_lit);
            let action_value = F::new(#action_value_lit);
            let mask = if price #cond_op cond_value { F::new(1.0f32) } else { F::new(0.0f32) };
            let result = price #action_op action_value;
            output[ABSOLUTE_POS] = result * mask;
        }
    }));

    TokenStream::from(quote!(#item_fn))
}
