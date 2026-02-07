use argh;
use argh::FromArgs;
use rten::Model;
use rten_text::Tokenizer;
use std::error::Error;
use std::io;
use std::io::{stdout, Write};
use std::time::Instant;

mod process;
mod prompt;

use crate::process::{do_one_iteration, generator_from_model};
use crate::prompt::{ChatConfig, VerseContext};

// Structure to handle command line arguments
#[derive(FromArgs)]
#[argh(description = "cli args")]
pub(crate) struct Args {
    #[argh(positional)]
    pub(crate) model: String,
    #[argh(positional)]
    pub(crate) tokenizer_config: String,
}

// Generate a CLI prompt that shows the state of chat options
fn do_cli_prompt(keep_history: bool, show_prompt: bool, show_time: bool) -> () {
    if keep_history {
        print!(">> +history ");
    } else {
        print!(">> -history ");
    }
    if show_prompt {
        print!("+prompt ");
    } else {
        print!("-prompt ");
    }
    if show_time {
        print!("+time >> ");
    } else {
        print!("-time >> ");
    }
    let _ = stdout().flush();
}

// Read a line of CLI input
fn read_input() -> String {
    let mut user_input_buffer = String::new();
    let _n_read = io::stdin()
        .read_line(&mut user_input_buffer)
        .expect("read_line");
    user_input_buffer
}

fn main() -> Result<(), Box<dyn Error>> {
    // Put all config into a structure
    let args: Args = argh::from_env();
    let mut config = ChatConfig {
        model_path: args.model,
        tokenizer_path: args.tokenizer_config,
        temperature: 0.5,
        top_k: 20,
        keep_history: false,
        show_prompt: false,
        show_time: true
    };

    // Set up model
    let model = unsafe { Model::load_mmap(config.model_path) }?;
    let tokenizer = Tokenizer::from_file(&config.tokenizer_path)?;
    let mut generator = generator_from_model(&model, &tokenizer, config.top_k, config.temperature);

    // Get RAG data
    let rag_json_path = std::path::PathBuf::from("./test_data/JHN/ch_3/v16.json");
    let absolute_rag_json_path = std::path::absolute(&rag_json_path).expect("absolute");
    let verse_context_string =
        std::fs::read_to_string(&absolute_rag_json_path).expect("Read verse context");
    let rag_json: VerseContext =
        serde_json::from_str(&verse_context_string).expect("Parse verse context");

    // 'Welcome' output
    println!("# Hello");
    println!("## Commands: /+history|-history|+prompt|-prompt|+time|-time|clear/");
    println!("## Empty line to quit");
    println!();
    println!("# Ask me a question about this verse!");
    loop {
        // Get input from user
        do_cli_prompt(config.keep_history, config.show_prompt, config.show_time);
        let user_input = read_input();

        // Handle special CLI strings
        if user_input.clone().trim().len() == 0 {
            println!("# Goodbye");
            return Ok(());
        }
        if user_input.clone().trim() == "/+history/" {
            config.keep_history = true;
            println!("# Keeping History");
            continue;
        }
        if user_input.clone().trim() == "/-history/" {
            config.keep_history = false;
            println!("# Not Keeping History");
            continue;
        }
        if user_input.clone().trim() == "/+prompt/" {
            config.show_prompt = true;
            println!("# Showing Prompt");
            continue;
        }
        if user_input.clone().trim() == "/-prompt/" {
            config.show_prompt = false;
            println!("# Not Showing Prompt");
            continue;
        }
        if user_input.clone().trim() == "/+time/" {
            config.show_time = true;
            println!("# Showing Time");
            continue;
        }
        if user_input.clone().trim() == "/-time/" {
            config.show_time = false;
            println!("# Not Showing Time");
            continue;
        }
        if user_input.clone().trim() == "/clear/" {
            if config.keep_history {
                generator = generator_from_model(&model, &tokenizer, config.top_k, config.temperature);
            }
            println!("# Cleared History");
            continue;
        }
        // Reset history if necessary
        if !config.keep_history {
            generator = generator_from_model(&model, &tokenizer, config.top_k, config.temperature);
        }
        // Process the input
        let now = Instant::now();
        let output_tokens = do_one_iteration(&mut generator, &tokenizer, rag_json.clone(), user_input, config.show_prompt.clone())?;
        if config.show_time {
            println!("# Processed in {:.2?} secs", now.elapsed());
        }
        // Collect output
        for output_token in output_tokens {
            print!("{}", output_token);
            stdout().flush().expect("flush after output token");
        }
        // And we're ready to do it again!
        println!();
    }
}
