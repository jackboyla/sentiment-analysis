
from transformers.models.canine import tokenization_canine
from transformers.models.canine.tokenization_canine import AddedToken
from collections import Counter

SPECIAL_CODEPOINT_NAMES = [

    "[CLS]",
    "[SEP]",
    "[BOS]",
    "[MASK]",
    "[RESERVED]",
    "[UNK]",

]


class CharTokenizer(tokenization_canine.CanineTokenizer, tokenization_canine.PreTrainedTokenizer):

    def __init__(self,
                 vocab_size,
                 train_sequences,
                 add_prefix_space=False,
                 model_max_length=2048,
                 **kwargs,):
        
        # Creates a mapping for looking up the IDs of special symbols.
        self._special_codepoints = {}
        self._special_codepoints["[PAD]"] = 0
        for idx, name in enumerate(SPECIAL_CODEPOINT_NAMES):
            self._special_codepoints[name] = idx+1

        # Creates a mapping for looking up the string forms of special symbol IDs.
        self._special_codepoint_strings = {
            codepoint: name for name, codepoint in self._special_codepoints.items()
        }

        bos_token = chr(self._special_codepoints['[BOS]'])
        eos_token = chr(self._special_codepoints['[SEP]'])
        sep_token = chr(self._special_codepoints['[SEP]'])
        cls_token = chr(self._special_codepoints['[CLS]'])
        pad_token = chr(self._special_codepoints['[PAD]'])
        unk_token = chr(self._special_codepoints['[UNK]'])
        mask_token = chr(self._special_codepoints['[MASK]'])
        
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        self.unk_token = unk_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        tokenization_canine.PreTrainedTokenizer.__init__(
            self,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            unk_token=unk_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kwargs,
        )

        self._num_special_tokens = len(self._special_codepoints)

        self.vocab_to_id = self.get_vocabulary(train_sequences, vocab_size)
        self._unicode_vocab_size = len(self.vocab_to_id)



    def get_vocabulary(self, sequences, vocab_size):
        counter = Counter()
        for text in sequences:
            characters = [char for char in text]
            counter.update(characters)
        most_common_chars = counter.most_common(vocab_size)
        # Add the most common characters to vocab; Assign index to each character
        # + num of special tokens
        vocabulary = {char: (idx + self._num_special_tokens) for idx, (char, _) in enumerate(most_common_chars)}
        # Add special codepoints to vocab
        vocabulary.update({chr(codepoint): codepoint for codepoint in self._special_codepoints.values()})
        return vocabulary
    

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (i.e. a Unicode character) in an id (i.e. its integer Unicode code point value)."""
        try:
            if token in self.vocab_to_id:
                return self.vocab_to_id[token]
            else:
                return self.vocab_to_id[self.unk_token]
        except TypeError:
            raise ValueError(f"invalid token: '{token}'")
        
    def _convert_id_to_token(self, index: int) -> str:
        """
        Converts a Unicode code point (integer) in a token (str). In case it's a special code point, convert to
        human-readable format.
        """
        try:
            if index in self._special_codepoint_strings:
                return self._special_codepoint_strings[index]
            return self.id_to_vocab[index]
        except TypeError:
            raise ValueError(f"invalid id: {index}")
