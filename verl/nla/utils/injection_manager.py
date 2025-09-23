"""Centralized injection token management for NLA."""

from typing import Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class InjectionTokenInfo:
    """Information about the injection token being used."""
    character: str
    token_id: int
    is_auto_selected: bool


class InjectionTokenManager:
    """Manages injection token selection and validation for NLA."""

    # Rarely-used Chinese characters for injection marking
    CANDIDATE_CHARS = [
        "㊗",  # Circled ideograph congratulation
        "㊙",  # Circled ideograph secret
        "㊚",  # Circled ideograph proper
        "㊛",  # Circled ideograph financial
        "㊜",  # Circled ideograph excellent
        "㊝",  # Circled ideograph print
        "㊞",  # Circled ideograph attention
        "㊟",  # Circled ideograph item
        "㊠",  # Circled ideograph rest
        "㊡",  # Circled ideograph copy
        "㊢",  # Circled ideograph night
        "㊣",  # Circled ideograph correct
        "㊤",  # Circled ideograph up
        "㊥",  # Circled ideograph middle
        "㊦",  # Circled ideograph down
        "㊧",  # Circled ideograph left
        "㊨",  # Circled ideograph right
        "㊩",  # Circled ideograph medical
        "㊪",  # Circled ideograph religion
        "㊫",  # Circled ideograph study
        "㊬",  # Circled ideograph supervise
        "㊭",  # Circled ideograph enterprise
        "㊮",  # Circled ideograph resource
        "㊯",  # Circled ideograph alliance
        "㊰",  # Circled ideograph night (alternate)
        "囍",  # Double happiness
        "囧",  # Jiong (embarrassed face)
        "囪",  # Window
        "囫",  # Whole
        "囬",  # Return variant
        "囮",  # Decoy
        "囯",  # Country variant
        "囱",  # Chimney
        "囲",  # Surround
        "図",  # Diagram
        "囶",  # Rare character
        "囷",  # Granary
        "囸",  # Day variant
        "囹",  # Prison
        "囻",  # Nation variant
    ]

    def __init__(self, tokenizer: Any, injection_token: Optional[str] = None):
        """
        Initialize the injection token manager.

        Args:
            tokenizer: The tokenizer to use
            injection_token: Optional specific token to use. If None, auto-selects.
        """
        self.tokenizer = tokenizer
        self.token_info = self._setup_injection_token(injection_token)

    def _setup_injection_token(self, requested_token: Optional[str]) -> InjectionTokenInfo:
        """
        Set up the injection token, either using the requested one or auto-selecting.

        Args:
            requested_token: The requested token, or None to auto-select

        Returns:
            InjectionTokenInfo with the selected token details

        Raises:
            RuntimeError: If no suitable token can be found
        """
        if requested_token is not None:
            # Use the requested token
            if requested_token not in self.tokenizer.get_vocab():
                raise RuntimeError(
                    f"Requested injection token '{requested_token}' not found in tokenizer vocabulary. "
                    "Please use an existing token or let the system auto-select one."
                )
            token_id = self.tokenizer.convert_tokens_to_ids(requested_token)

            # Validate it works correctly
            if not self._validate_token(requested_token, token_id):
                raise RuntimeError(
                    f"Token '{requested_token}' does not meet requirements for injection. "
                    "It must tokenize to a single token and preserve structure in <concept> tags."
                )

            return InjectionTokenInfo(
                character=requested_token,
                token_id=token_id,
                is_auto_selected=False
            )
        else:
            # Auto-select a suitable token
            char, token_id = self._find_suitable_token()
            if char is None:
                raise RuntimeError(
                    "Could not find a suitable injection token in the tokenizer vocabulary. "
                    "This tokenizer may not support the required character set."
                )

            print(f"Auto-selected '{char}' (token ID: {token_id}) as injection marker")
            return InjectionTokenInfo(
                character=char,
                token_id=token_id,
                is_auto_selected=True
            )

    def _find_suitable_token(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Find a suitable rare token for injection marking.

        Returns:
            tuple: (character, token_id) of the suitable token, or (None, None) if not found
        """
        for char in self.CANDIDATE_CHARS:
            # Test if this character meets our requirements
            char_tokens = self.tokenizer.encode(char, add_special_tokens=False)

            # Must tokenize to exactly one token
            if len(char_tokens) != 1:
                continue

            token_id = char_tokens[0]

            # Validate it works in context
            if self._validate_token(char, token_id):
                return char, token_id

        return None, None

    def _validate_token(self, char: str, token_id: int) -> bool:
        """
        Validate that a character works correctly as an injection token.

        Args:
            char: The character to validate
            token_id: The expected token ID

        Returns:
            bool: True if valid, False otherwise
        """
        # Test 1: Single character tokenization
        char_tokens = self.tokenizer.encode(char, add_special_tokens=False)
        if len(char_tokens) != 1 or char_tokens[0] != token_id:
            return False

        # Test 2: Within concept tags
        test_text = f"<concept>{char}</concept>"
        tokens = self.tokenizer.encode(test_text, add_special_tokens=False)

        # The character should appear exactly once
        if tokens.count(token_id) != 1:
            return False

        # Test 3: Decode preserves structure
        decoded = self.tokenizer.decode(tokens)
        if decoded != test_text:
            return False

        # Test 4: Character is preserved when decoded alone
        single_decoded = self.tokenizer.decode([token_id])
        if char not in single_decoded:
            return False

        return True

    def validate_text_with_injection(self, text: str) -> bool:
        """
        Validate that text with injection markers is properly formatted.

        Args:
            text: The text to validate

        Returns:
            bool: True if properly formatted, False otherwise
        """
        # Check that injection tokens only appear within <concept> tags
        import re

        # Find all occurrences of the injection character outside of concept tags
        pattern = f"<concept>{re.escape(self.token_info.character)}</concept>"

        # Remove all properly tagged occurrences
        text_without_tags = re.sub(pattern, "", text)

        # The injection character should not appear in the remaining text
        if self.token_info.character in text_without_tags:
            return False

        return True

    def prepare_text_for_injection(self, text: str, injection_position: str = "end") -> str:
        """
        Prepare text by adding injection marker at the specified position.

        Args:
            text: The original text
            injection_position: Where to add the marker ("start", "end", or "manual")

        Returns:
            str: Text with injection marker added
        """
        if injection_position == "manual":
            # Assume the text already has markers
            return text

        marker = f"<concept>{self.token_info.character}</concept>"

        if injection_position == "start":
            return marker + " " + text
        elif injection_position == "end":
            return text + " " + marker
        else:
            raise ValueError(f"Invalid injection_position: {injection_position}")

    @property
    def character(self) -> str:
        """Get the injection character being used."""
        return self.token_info.character

    @property
    def token_id(self) -> int:
        """Get the token ID of the injection character."""
        return self.token_info.token_id

    @property
    def is_auto_selected(self) -> bool:
        """Check if the token was auto-selected."""
        return self.token_info.is_auto_selected