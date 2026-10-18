//! Dory chat encoding round trip. Needs `DORY_TOKENIZER=<path to tokenizer.json>`; skipped otherwise.

use std::sync::Arc;

use hanashi::{
    Encoding as _,
    chat::hanashi::{HanashiEncodingImpl, config::HanashiConfig},
};
use shoji::types::session::chat::{ChatContentBlock, ChatMessage, ChatRole};
use tokenizers::Tokenizer;

fn encoding() -> Option<(HanashiEncodingImpl, Arc<Tokenizer>)> {
    let path = std::env::var("DORY_TOKENIZER").ok()?;
    let tokenizer = Arc::new(Tokenizer::from_file(path).unwrap());
    Some((HanashiEncodingImpl::new(HanashiConfig::Dory, tokenizer.clone()).unwrap(), tokenizer))
}

#[test]
fn dory_encodes_prompt_and_decodes_reply() {
    let Some((mut encoding, tokenizer)) = encoding() else {
        return;
    };
    encoding.encode(vec![ChatMessage::user().with_text("Hello".to_string())]).unwrap();
    assert_eq!(
        encoding.state().text(),
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    for reply in ["Hi there!<|im_end|>", "</think>Hi there!<|im_end|>"] {
        encoding.reset().unwrap();
        encoding.encode(vec![ChatMessage::user().with_text("Hello".to_string())]).unwrap();
        for token_id in tokenizer.encode(reply, false).unwrap().get_ids() {
            if let Err(error) = encoding.decode(vec![*token_id]) {
                panic!("decode failed at token {token_id} of {reply:?}: {error:?}");
            }
        }
        let message = encoding.state().messages.last().unwrap();
        assert_eq!(message.role, ChatRole::Assistant {});
        println!("{reply:?} -> {:?}", message.content);
        // With thinking on (the default), the reply is reasoning until `</think>`.
        let has_text = message.content.iter().any(|block| matches!(block, ChatContentBlock::Text { .. }));
        assert_eq!(has_text, reply.starts_with("</think>"));
    }
}
