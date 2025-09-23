"""Simple test for injection token management without downloading models."""

import sys
sys.path.append('/Users/kit/Documents/Anthropic/NLA/verl')

from unittest.mock import Mock
from verl.nla.utils.injection_manager import InjectionTokenManager


def create_mock_tokenizer():
    """Create a mock tokenizer that mimics real tokenizer behavior."""
    tokenizer = Mock()

    # Create a simple vocabulary
    vocab = {
        "hello": 1,
        "world": 2,
        "㊗": 100,  # Our Chinese character
        "<": 10,
        "concept": 11,
        ">": 12,
        "/": 13,
    }

    tokenizer.get_vocab = Mock(return_value=vocab)

    def mock_encode(text, add_special_tokens=False):
        """Simple mock encoding."""
        if text == "㊗":
            return [100]
        elif text == "<concept>㊗</concept>":
            return [10, 11, 12, 100, 10, 13, 11, 12]  # Simplified encoding
        elif text == "!":
            return [50]  # Some other token
        else:
            # Return something reasonable
            return [1, 2]  # Just return some tokens

    def mock_decode(tokens):
        """Simple mock decoding."""
        if tokens == [100]:
            return "㊗"
        elif tokens == [10, 11, 12, 100, 10, 13, 11, 12]:
            return "<concept>㊗</concept>"
        else:
            return "decoded text"

    def mock_convert_tokens_to_ids(token):
        """Convert token to ID."""
        return vocab.get(token, -1)

    tokenizer.encode = Mock(side_effect=mock_encode)
    tokenizer.decode = Mock(side_effect=mock_decode)
    tokenizer.convert_tokens_to_ids = Mock(side_effect=mock_convert_tokens_to_ids)

    return tokenizer


def test_auto_select():
    """Test auto-selection of injection token."""
    tokenizer = create_mock_tokenizer()

    manager = InjectionTokenManager(tokenizer)

    # Should select the Chinese character
    assert manager.character == "㊗"
    assert manager.token_id == 100
    assert manager.is_auto_selected

    print(f"✅ Auto-selected '{manager.character}' (ID: {manager.token_id})")


def test_specific_token():
    """Test using a specific token."""
    tokenizer = create_mock_tokenizer()

    # Mock for "hello" token
    def mock_encode_hello(text, add_special_tokens=False):
        if text == "hello":
            return [1]
        elif text == "<concept>hello</concept>":
            return [10, 11, 12, 1, 10, 13, 11, 12]
        return [999]

    def mock_decode_hello(tokens):
        if tokens == [1]:
            return "hello"
        elif tokens == [10, 11, 12, 1, 10, 13, 11, 12]:
            return "<concept>hello</concept>"
        return "other"

    tokenizer.encode = Mock(side_effect=mock_encode_hello)
    tokenizer.decode = Mock(side_effect=mock_decode_hello)

    manager = InjectionTokenManager(tokenizer, "hello")

    assert manager.character == "hello"
    assert manager.token_id == 1
    assert not manager.is_auto_selected

    print(f"✅ Specific token '{manager.character}' (ID: {manager.token_id})")


def test_invalid_token():
    """Test that invalid tokens are rejected."""
    tokenizer = create_mock_tokenizer()

    try:
        manager = InjectionTokenManager(tokenizer, "invalid_token")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not found in tokenizer vocabulary" in str(e)
        print("✅ Invalid token properly rejected")


def test_text_preparation():
    """Test text preparation methods."""
    tokenizer = create_mock_tokenizer()
    manager = InjectionTokenManager(tokenizer)

    text = "Test text"

    # Test end injection
    end_text = manager.prepare_text_for_injection(text, "end")
    assert end_text == f"Test text <concept>{manager.character}</concept>"

    # Test start injection
    start_text = manager.prepare_text_for_injection(text, "start")
    assert start_text == f"<concept>{manager.character}</concept> Test text"

    print("✅ Text preparation works correctly")


def test_validation():
    """Test text validation."""
    tokenizer = create_mock_tokenizer()
    manager = InjectionTokenManager(tokenizer)

    # Valid text
    valid = f"Text <concept>{manager.character}</concept> more"
    assert manager.validate_text_with_injection(valid)

    # Invalid text - character outside tags
    invalid = f"Text {manager.character} more"
    assert not manager.validate_text_with_injection(invalid)

    print("✅ Text validation works correctly")


if __name__ == "__main__":
    test_auto_select()
    test_specific_token()
    test_invalid_token()
    test_text_preparation()
    test_validation()

    print("\n🎉 All injection manager tests passed!")