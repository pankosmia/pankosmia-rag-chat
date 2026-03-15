use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use rten_text::{Tokenizer, TokenizerError};
use crate::process::MessageChunk;

pub struct ChatConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub rag_json_path: String,
    pub temperature: f32,
    pub top_k: usize,
    pub keep_history: bool,
    pub show_prompt: bool,
    pub show_time: bool
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VersePrompts {
    pub system: String,
    pub prologue: String,
    pub user_prologue: String,
    pub juxta: String,
    pub notes: String,
    pub snippets: String,
    pub translations: String
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerseContext {
    pub juxta: String,
    pub translations: BTreeMap<String, String>,
    pub notes: BTreeMap<String, Vec<String>>,
    pub snippets: BTreeMap<String, Vec<String>>,
    pub prompts: VersePrompts
}

pub(crate) fn encode_system_message(tokenizer: &Tokenizer, system_prompt: String) -> Result<Vec<u32>, TokenizerError> {
    encode_message(
        tokenizer,
        format!("system\n{}", &system_prompt)
    )
}
pub(crate) fn encode_message(
    tokenizer: &Tokenizer,
    user_prompt: String,
) -> Result<Vec<u32>, TokenizerError> {
    let im_start_token = tokenizer.get_token_id("<|im_start|>")?;
    let im_end_token = tokenizer.get_token_id("<|im_end|>")?;

    let mut end_of_turn_tokens = Vec::new();
    end_of_turn_tokens.push(im_end_token);

    if let Ok(end_of_text_token) = tokenizer.get_token_id("<|endoftext|>") {
        end_of_turn_tokens.push(end_of_text_token);
    }

    let mut token_ids = Vec::new();
    let chunks = &[
        MessageChunk::Token(im_start_token),
        MessageChunk::Text("user\n"),
        MessageChunk::Text(&user_prompt),
        MessageChunk::Token(im_end_token),
        MessageChunk::Text("\n"),
        MessageChunk::Token(im_start_token),
        MessageChunk::Text("assistant\n"),
    ];
    for chunk in chunks {
        match chunk {
            MessageChunk::Token(tok_id) => token_ids.push(*tok_id),
            MessageChunk::Text(text) => {
                let encoded = tokenizer.encode(*text, None)?;
                token_ids.extend(encoded.token_ids());
            }
        }
    }
    Ok(token_ids)
}


pub(crate) fn generate_user_prompt(
    _bcv: String,
    printable_bcv: String,
    rag_json: VerseContext,
    user_input: String,
) -> String {
    let mut translation_contexts: Vec<String> = Vec::new();
    for (k, v) in rag_json.translations {
        translation_contexts.push(format!("\n- {} ({}): {}\n", &printable_bcv, &k, &v));
    }
    let translation_context: String = translation_contexts.into_iter().collect();
    let translations_prompt;
    if translation_context.len() > 0 {
        translations_prompt = format!("{}{}\n\n", &rag_json.prompts.translations, translation_context);
    } else {
        translations_prompt = "".to_string();
    }


    let juxta_context = rag_json.juxta.clone();
    let juxta_prompt;
    if juxta_context.len() > 0 {
        juxta_prompt = format!("{}\n\n{}\n\n", &rag_json.prompts.juxta, juxta_context);
    } else {
        juxta_prompt = "".to_string();
    }

    let mut note_contexts: Vec<String> = Vec::new();
    for (k, v) in rag_json.notes {
        let mut numbered_notes: Vec<String> = Vec::new();
        let mut note_n = 1;
        for note in v {
            numbered_notes.push(format!("({}) {} ", note_n, &note));
            note_n += 1;
        }
        let numbered_note_string: String = numbered_notes.into_iter().collect();
        note_contexts.push(format!(
            "\n- {} from the {}: {}\n",
            &printable_bcv, &k, &numbered_note_string
        ));
    }
    let note_context: String = note_contexts.into_iter().collect();
    let note_prompt;
    if note_context.len() > 0 {
        note_prompt = format!("{}{}\n\n",&rag_json.prompts.notes, &note_context);
    } else {
        note_prompt = "".to_string();
    }

    let mut snippet_contexts: Vec<String> = Vec::new();
    for (snippet_key, snippet_value) in rag_json.snippets {
        let mut snippet_notes: Vec<String> = Vec::new();
        let mut note_n = 1;
        for note in snippet_value {
            snippet_notes.push(format!("({}) {} ", note_n, &note));
            note_n += 1;
        }
        let numbered_note_string: String = snippet_notes.into_iter().collect();
        snippet_contexts.push(format!(
            "\n- the word or words '{}' in {}: {}\n",
            &snippet_key, &printable_bcv, &numbered_note_string
        ));
    }
    let snippet_context: String = snippet_contexts.into_iter().collect();
    let snippet_prompt;
    if snippet_context.len() > 0 {
        snippet_prompt = format!("{}\n{}\n\n", &rag_json.prompts.snippets, &snippet_context);
    } else {
        snippet_prompt = "".to_string();
    }

    format!(
        "{}\n\n{}{}{}{}{}\n\n**{}**",
        &rag_json.prompts.prologue,
        &juxta_prompt,
        &translations_prompt,
        &note_prompt,
        &snippet_prompt,
        &rag_json.prompts.user_prologue,
        &user_input.trim()
    )
}