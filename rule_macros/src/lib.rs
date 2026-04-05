use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitFloat, LitStr};

#[derive(serde::Deserialize)]
struct RuleConfig {
    #[serde(rename = "if")]
    if_: RuleIf,
    then: ActConfig,
}

#[derive(serde::Deserialize)]
struct RuleIf {
    field: String,
    #[serde(default)]
    gte: Option<ValueConfig>,
    #[serde(default)]
    lte: Option<ValueConfig>,
}

#[derive(serde::Deserialize)]
struct ValueConfig {
    value: f32,
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
    let cfg: Vec<RuleConfig> = serde_json::from_str(&json).unwrap_or_else(|err| {
        panic!(
            "include_rule_kernel: invalid JSON in {}: {}",
            full_path.display(),
            err
        )
    });

    let mut rule_blocks: Vec<proc_macro2::TokenStream> = Vec::new();

    for rule in cfg {
        let (cond_op, cond_value) = match (rule.if_.gte, rule.if_.lte) {
            (Some(v), None) => (quote!(>), v.value),
            (None, Some(v)) => (quote!(<), v.value),
            _ => panic!("include_rule_kernel: condition must have exactly one of gte/lte"),
        };

        let field = rule.if_.field.to_lowercase();
        let field_expr = match field.as_str() {
            "price" => quote!(price),
            "quantity" => quote!(quantity),
            _ => panic!("include_rule_kernel: unsupported field {}", field),
        };

        let (action_op, action_value) = match rule.then {
            ActConfig::Plus { value } => (quote!(+), value),
            ActConfig::Multiply { value } => (quote!(*), value),
        };

        let cond_value_lit =
            LitFloat::new(&format!("{cond_value}f32"), proc_macro2::Span::call_site());
        let action_value_lit =
            LitFloat::new(&format!("{action_value}f32"), proc_macro2::Span::call_site());

        rule_blocks.push(quote! {
            let cond_value = F::new(#cond_value_lit);
            let action_value = F::new(#action_value_lit);
            if #field_expr #cond_op cond_value {
                value = value #action_op action_value;
                bonus = value;
            }
        });
    }

    let mut item_fn = parse_macro_input!(item as ItemFn);

    item_fn.block = Box::new(syn::parse_quote!({
        const _: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #path_lit));
        if ABSOLUTE_POS < prices.len() {
            let price = prices[ABSOLUTE_POS];
            let quantity = quantities[ABSOLUTE_POS];
            let mut value = price;
            let mut bonus = F::new(0.0f32);
            #(#rule_blocks)*
            output[ABSOLUTE_POS] = bonus;
        }
    }));

    TokenStream::from(quote!(#item_fn))
}
