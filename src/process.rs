use rten::Model;
use rten_generate::filter::Chain;
use rten_generate::sampler::Multinomial;
use rten_generate::{Generator, GeneratorUtils};
use rten_text::{TokenId, Tokenizer};
use std::error::Error;

use crate::prompt::{encode_message, encode_system_message, generate_user_prompt, VerseContext};
pub(crate) enum MessageChunk<'a> {
    Text(&'a str),
    Token(u32),
}

pub(crate) fn get_end_of_turn_tokens(tokenizer: &Tokenizer) -> Vec<TokenId> {
    let im_end_token = tokenizer
        .get_token_id("<|im_end|>")
        .expect("get token id for end");

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }
    end_of_turn_tokens
}

pub(crate) fn generator_from_model<'a>(
    model: &'a Model,
    tokenizer: &'a Tokenizer,
    top_k: usize,
    temperature: f32,
) -> Generator<'a> {
    let prompt = encode_system_message(tokenizer).expect("encode system message");
    Generator::from_model(model)
        .expect("generator from model")
        .with_prompt(&prompt)
        .with_logits_filter(Chain::new().top_k(top_k).temperature(temperature))
        .with_sampler(Multinomial::new())
}

pub(crate) fn do_one_iteration(
    generator: &mut Generator,
    tokenizer: &Tokenizer,
    rag_json: VerseContext,
    user_input: String,
    show_prompt: bool
) -> Result<Vec<String>, Box<dyn Error>> {
    let end_of_turn_tokens = get_end_of_turn_tokens(tokenizer);

    let user_text =
        generate_user_prompt("JHN 3:16".to_string(), "John 3:16".to_string(), rag_json, user_input);
    let token_ids = encode_message(&tokenizer, user_text.clone())?;

    generator.append_prompt(&token_ids);

    let decoder = generator
        .by_ref()
        .stop_on_tokens(&end_of_turn_tokens)
        .decode(&tokenizer);
    let mut tokens = Vec::new();
    if show_prompt {
        tokens.push(format!("\n# Prompt\n\n{}\n", &user_text));
    }
    for token in decoder {
        let token = token?;
            tokens.push(format!("{}", token));
        };
    Ok(tokens)
}
