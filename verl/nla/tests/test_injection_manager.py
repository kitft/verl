"""Test injection token management."""

import torch
from transformers import AutoTokenizer
from verl.nla.utils.injection_manager import InjectionTokenManager


def test_auto_select_injection_token():
    """Test that we can auto-select a suitable injection token."""
    # Use a real tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create manager with auto-selection
    manager = InjectionTokenManager(tokenizer)

    print(f"✅ Auto-selected token: '{manager.character}' (ID: {manager.token_id})")

    # Verify the token exists in vocabulary
    assert manager.character in tokenizer.get_vocab()

    # Verify it tokenizes to a single token
    tokens = tokenizer.encode(manager.character, add_special_tokens=False)
    assert len(tokens) == 1
    assert tokens[0] == manager.token_id

    print("✅ Token exists in vocabulary and tokenizes correctly")


def test_injection_in_concept_tags():
    """Test that injection tokens work correctly in <concept> tags."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    manager = InjectionTokenManager(tokenizer)

    # Test text with concept tags
    test_text = f"<concept>{manager.character}</concept>"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)

    # The injection token should appear exactly once
    assert tokens.count(manager.token_id) == 1

    # Decode should preserve structure
    decoded = tokenizer.decode(tokens)
    assert decoded == test_text

    print(f"✅ Token works correctly in concept tags: {test_text}")


def test_text_preparation():
    """Test text preparation with injection markers."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    manager = InjectionTokenManager(tokenizer)

    original_text = "This is a test prompt"

    # Test end injection
    end_text = manager.prepare_text_for_injection(original_text, "end")
    expected_end = f"This is a test prompt <concept>{manager.character}</concept>"
    assert end_text == expected_end

    # Test start injection
    start_text = manager.prepare_text_for_injection(original_text, "start")
    expected_start = f"<concept>{manager.character}</concept> This is a test prompt"
    assert start_text == expected_start

    print("✅ Text preparation works correctly")


def test_validation():
    """Test validation of text with injection markers."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    manager = InjectionTokenManager(tokenizer)

    # Valid text - injection only in concept tags
    valid_text = f"Some text <concept>{manager.character}</concept> more text"
    assert manager.validate_text_with_injection(valid_text)

    # Invalid text - injection outside concept tags
    invalid_text = f"Some text {manager.character} more text"
    assert not manager.validate_text_with_injection(invalid_text)

    # Invalid text - injection both inside and outside tags
    mixed_text = f"Text {manager.character} and <concept>{manager.character}</concept>"
    assert not manager.validate_text_with_injection(mixed_text)

    print("✅ Text validation works correctly")


def test_multiple_injections():
    """Test handling multiple injection points."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    manager = InjectionTokenManager(tokenizer)

    # Text with multiple injection points
    text = f"First <concept>{manager.character}</concept> middle <concept>{manager.character}</concept> end"
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Should have exactly 2 injection tokens
    assert tokens.count(manager.token_id) == 2

    # Decode should preserve structure
    decoded = tokenizer.decode(tokens)
    assert decoded == text

    print("✅ Multiple injection points handled correctly")


def test_no_tokenizer_modification():
    """Verify that we don't modify the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Get original vocab size
    original_vocab_size = len(tokenizer.get_vocab())

    # Create manager - should not add tokens
    manager = InjectionTokenManager(tokenizer)

    # Vocab size should be unchanged
    assert len(tokenizer.get_vocab()) == original_vocab_size

    print("✅ Tokenizer vocabulary unchanged")


def test_specific_token_request():
    """Test using a specific requested token."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Request a specific character that should exist
    # Use a character we know is in GPT-2's vocab
    specific_char = "!"  # Exclamation mark is definitely in vocab

    manager = InjectionTokenManager(tokenizer, specific_char)

    assert manager.character == specific_char
    assert not manager.is_auto_selected

    print(f"✅ Specific token request works: '{specific_char}'")


def test_invalid_token_request():
    """Test that invalid token requests raise errors."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Request a token that definitely won't exist
    invalid_token = "🦄🦄🦄INVALID🦄🦄🦄"

    try:
        manager = InjectionTokenManager(tokenizer, invalid_token)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not found in tokenizer vocabulary" in str(e)
        print("✅ Invalid token request properly rejected")


if __name__ == "__main__":
    test_auto_select_injection_token()
    test_injection_in_concept_tags()
    test_text_preparation()
    test_validation()
    test_multiple_injections()
    test_no_tokenizer_modification()
    test_specific_token_request()
    test_invalid_token_request()

    print("\n🎉 All injection manager tests passed!")