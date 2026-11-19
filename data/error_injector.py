import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class InjectedError:
    error_type: str # "spelling" | "ocr_symbol" | "ocr_formula" | "variable" | "handwriting_confusion"
    original_span: str
    corrupted_span: str
    char_start: int
    char_end: int


# OCR confusion tables

OCR_SYMBOL_CONFUSIONS: Dict[str, List[str]] = {
    r"\leq": [r"<", r"\le", r"\geq"],
    r"\geq": [r">", r"\ge", r"\leq"],
    r"\neq": [r"=", r"\ne", r"\not="],
    r"\in": [r"\ni", r"\notin", r"\\in", r"E"],
    r"\forall": [r"\exists", r"A", r"\lor all"],
    r"\exists": [r"\forall", r"E", r"\exist"],
    r"\subset": [r"\supset", r"\subseteq", r"C"],
    r"\subseteq": [r"\supseteq", r"\subset"],
    r"\to": [r"\rightarrow", r"\\to", r"-"],
    r"\infty": [r"\infinity", r"oo", r"\inf ty"],
    r"\pm": [r"+", r"\mp"],
    r"\times": [r"x", r"\time", r"X"],
    r"\cdot": [r".", r"\dot", r"\cot"],
    r"\cup": [r"\cap", r"U", r"\Cup"],
    r"\cap": [r"\cup", r"n", r"\Cap"],
    r"\int": [r"f", r"\nt", r"\iint"],
    r"\sum": [r"\Sigma", r"\\sum", r"\Sum"],
    r"\prod": [r"\Pi", r"\product"],
    r"\partial": [r"\delta", r"d", r"\pratial"],
    r"\nabla": [r"\Delta", r"\nable"],
    r"\alpha": [r"a", r"\alph", r"\Alpha"],
    r"\beta": [r"\Beta", r"B", r"\bata"],
    r"\gamma": [r"\Gamma", r"y", r"\gama"],
    r"\delta": [r"\Delta", r"d", r"\dleta"],
    r"\epsilon": [r"\varepsilon", r"e", r"\epslon"],
    r"\theta": [r"\Theta", r"0", r"\thea"],
    r"\lambda": [r"\Lambda", r"A", r"\lamda"],
    r"\mu": [r"u", r"\nu", r"\Mu"],
    r"\pi": [r"\Pi", r"n", r"\p i"],
    r"\sigma": [r"\Sigma", r"o", r"\sgima"],
    r"\phi": [r"\Phi", r"\varphi", r"\psi"],
    r"\omega": [r"\Omega", r"w", r"\omga"],
}

OCR_FORMULA_CONFUSIONS: Dict[str, List[str]] = {
    r"^{2}": [r"^2", r"2", r"^{z}"],
    r"^{3}": [r"^3", r"3", r"^{s}"],
    r"^{n}": [r"^n", r"n", r"^{m}"],
    r"_{i}": [r"_i", r"i", r"_{1}"],
    r"_{n}": [r"_n", r"n", r"_{m}"],
    r"\frac{": [r"\fac{", r"\frac {", r"\\frac{"],
    r"\sqrt{": [r"\sqr{", r"\sqrt {", r"\\sqrt{", r"\sqrtf"],
    r"\mathbb{R}": [r"\mathbb{r}", r"R", r"\mabb{R}", r"\mathbf{R}"],
    r"\mathbb{Z}": [r"\mathbb{z}", r"Z", r"\mabb{Z}"],
    r"\mathbb{N}": [r"\mathbb{n}", r"N", r"\mabb{N}"],
    r"\mathbb{C}": [r"\mathbb{c}", r"C", r"\mabb{C}"],
    r"\lim_{": [r"\lim_", r"lim_{", r"\liim_{"],
    r"\sum_{": [r"\sum_", r"sum_{", r"\ssum_{"],
    r"\int_{": [r"\int_", r"int_{", r"\iint_{"],
    r"\log": [r"\lg", r"log", r"\lo g"],
    r"\sin": [r"sin", r"\sn", r"\sin "],
    r"\cos": [r"cos", r"\cs", r"\cos "],
    r"\tan": [r"tan", r"\tn", r"\tan "],
    r"\exp": [r"exp", r"\ex", r"\exp "],
}

MATH_SPELLING_WORDS: Dict[str, List[str]] = {
    "theorem": ["theroem", "theorom", "theorm", "thereom"],
    "lemma": ["lema", "lemm", "lamma"],
    "corollary": ["corallary", "corolary", "corollery"],
    "proposition": ["propostion", "propositon", "propsition"],
    "definition": ["defintion", "defnition", "definiton"],
    "continuous": ["continous", "continuos", "continious"],
    "differentiable": ["diferentiable", "differentiabel", "differntable"],
    "convergent": ["convergant", "convegent", "convergnet"],
    "divergent": ["divergant", "divegent", "divergnet"],
    "monotonically": ["monotonically", "monotoniclly", "montonically"],
    "injective": ["injecive", "injectve", "injcetive"],
    "surjective": ["surjecive", "surjectve", "surjcetive"],
    "bijective": ["bijecive", "bijectve", "bijcetive"],
    "homeomorphism": ["homeomorphsim", "homoemorphism", "homeomorphim"],
    "isomorphism": ["isomorphsim", "isomrphism", "isomorphim"],
    "polynomial": ["polynomal", "polynominal", "polynomail"],
    "eigenvalue": ["eignvalue", "eigenvaule", "eigenvlaue"],
    "eigenvector": ["eignvector", "eigenvctor", "eigenvecor"],
    "orthogonal": ["orthagonal", "orthoganal", "orthognal"],
    "symmetric": ["symetric", "symmetirc", "symmertic"],
    "determinant": ["determinat", "determiant", "detarminant"],
    "derivative": ["deriviative", "derivatve", "dervative"],
    "integral": ["intgral", "intergal", "integeral"],
    "bounded": ["bouned", "boundeed", "bunded"],
    "compact": ["compcat", "comapct", "compac"],
    "topology": ["topolgy", "toplogy", "topoplogy"],
    "sequence": ["sequnce", "seqeunce", "sequece"],
    "subsequence": ["subsequnce", "subseqeunce", "subequence"],
    "hypothesis": ["hypotheis", "hypotheisis", "hypothsis"],
    "sufficient": ["sufficent", "sufficiant", "sufficeint"],
    "necessary": ["neccessary", "necesary", "neccesary"],
    "inequality": ["inequalty", "ineqality", "inequailty"],
    "equation": ["equaton", "equaiton", "eqation"],
    "function": ["funciton", "fucntion", "funtion"],
    "variable": ["varaible", "variabel", "varialbe"],
    "matrix": ["matrx", "matirx", "matix"],
    "dimension": ["dimesion", "dimensoin", "dimenison"],
    "manifold": ["manfold", "maniflod", "manifod"],
    "subspace": ["subpsace", "subsapce", "subspce"],
    "neighborhood": ["neigborhood", "neighbourhod", "neighboorhood"],
    "countable": ["countble", "coutnable", "coutable"],
    "measurable": ["measurble", "measuable", "measurbale"],
    "integrable": ["integrble", "intgrable", "integable"],
    "holomorphic": ["holomrphic", "holomorphc", "holomorphci"],
    "analytic": ["anlaytic", "anlytic", "analyitc"],
    "commutative": ["commutaive", "comutative", "commutitve"],
    "associative": ["assocaitive", "associtave", "associtive"],
    "distributive": ["distributve", "distrbutive", "distribtive"],
}

HANDWRITING_CONFUSIONS: Dict[str, List[str]] = {
    "1": ["l", "I", "7"],
    "l": ["1", "I"],
    "I": ["1", "l"],
    "0": ["O", "o", "Q", "6", "8"],
    "O": ["0", "Q", "o"],
    "o": ["0", "O"],
    "Q": ["0", "O"],
    "2": ["Z", "z"],
    "Z": ["2"],
    "z": ["2"],
    "5": ["S", "s", "3"],
    "S": ["5", "s"],
    "s": ["5", "S"],
    "8": ["B", "0"],
    "B": ["8"],
    "9": ["q", "g"],
    "q": ["9", "g"],
    "g": ["9", "q", "y"],
    "7": ["1", "T"],
    "T": ["7"],
    "6": ["G", "0"],
    "G": ["6", "C"],
    "4": ["A"],
    "A": ["4"],
    "C": ["G"],
    "u": ["n", "v"],
    "n": ["u"],
    "U": ["V"],
    "V": ["U"],
    "i": ["j", "1"],
    "j": ["i"],
    "y": ["g"],
    "a": ["o"],
    "c": ["e"],
    "e": ["c"],
}

HANDWRITING_MIN_WORD_LENGTH = 4
HANDWRITING_WORD_PATTERN = re.compile(
    rf"\b[A-Za-z]{{{HANDWRITING_MIN_WORD_LENGTH},}}\b"
)
LATEX_MATH_ENVIRONMENTS = {
    "math",
    "displaymath",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
}

COMMON_VARIABLES = ["x", "y", "z", "t", "n", "m", "k", "i", "j", "a", "b", "c", "f", "g", "h"]


class ErrorInjector:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject_errors(
        self,
        text: str,
        num_errors: int = 2,
        error_types: Optional[List[str]] = None,
        error_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, List[InjectedError]]:
        if error_types is None:
            error_types = ["spelling", "ocr_symbol", "ocr_formula", "variable", "handwriting_confusion"]
        if error_weights is None:
            error_weights = {
                "spelling": 0.25,
                "ocr_symbol": 0.25,
                "ocr_formula": 0.20,
                "variable": 0.05,
                "handwriting_confusion": 0.25,
            }

        weights = [error_weights.get(t, 0.25) for t in error_types]
        chosen_types = self.rng.choices(error_types, weights=weights, k=num_errors)

        injected_errors: List[InjectedError] = []
        corrupted = text

        for etype in chosen_types:
            result = self._inject_single_error(corrupted, etype, injected_errors)
            if result is not None:
                corrupted, error = result
                self._shift_existing_offsets_after_edit(
                    existing=injected_errors,
                    edit_start=error.char_start,
                    original_len=len(error.original_span),
                    replacement_len=len(error.corrupted_span),
                )
                injected_errors.append(error)

        return corrupted, injected_errors

    def _inject_single_error(
        self,
        text: str,
        error_type: str,
        existing_errors: List[InjectedError],
    ) -> Optional[Tuple[str, InjectedError]]:
        if error_type == "spelling":
            return self._inject_spelling_error(text, existing_errors)
        elif error_type == "ocr_symbol":
            return self._inject_ocr_symbol_error(text, existing_errors)
        elif error_type == "ocr_formula":
            return self._inject_ocr_formula_error(text, existing_errors)
        elif error_type == "variable":
            return self._inject_variable_error(text, existing_errors)
        elif error_type == "handwriting_confusion":
            return self._inject_handwriting_confusion_error(text, existing_errors)
        return None

    def _overlaps_existing(
        self, start: int, end: int, existing: List[InjectedError]
    ) -> bool:
        for e in existing:
            if start < e.char_end and end > e.char_start:
                return True
        return False

    def _is_isolated_token(self, text: str, start: int, end: int) -> bool:
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        return (not left.isalnum()) and (not right.isalnum())

    def _find_closing_math_delimiter(
        self, text: str, start: int, delimiter: str
    ) -> int:
        i = start
        while i < len(text):
            if text[i] == "\\" and i + 1 < len(text):
                i += 2
                continue
            if delimiter == "$$":
                if text.startswith("$$", i):
                    return i
            elif delimiter == "$":
                if text[i] == "$" and not text.startswith("$$", i):
                    return i
            i += 1
        return -1

    def _find_latex_math_spans(self, text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        i = 0

        while i < len(text):
            env_match = re.match(r"\\begin\{([a-zA-Z*]+)\}", text[i:])
            if env_match:
                env_name = env_match.group(1)
                if env_name in LATEX_MATH_ENVIRONMENTS:
                    end_marker = rf"\end{{{env_name}}}"
                    end = text.find(end_marker, i + len(env_match.group(0)))
                    if end != -1:
                        span_end = end + len(end_marker)
                        spans.append((i, span_end))
                        i = span_end
                        continue

            if text.startswith(r"\(", i):
                end = text.find(r"\)", i + 2)
                if end != -1:
                    spans.append((i, end + 2))
                    i = end + 2
                    continue

            if text.startswith(r"\[", i):
                end = text.find(r"\]", i + 2)
                if end != -1:
                    spans.append((i, end + 2))
                    i = end + 2
                    continue

            if text[i] == "\\" and i + 1 < len(text):
                i += 2
                continue

            if text.startswith("$$", i):
                end = self._find_closing_math_delimiter(text, i + 2, "$$")
                if end != -1:
                    spans.append((i, end + 2))
                    i = end + 2
                    continue

            if text[i] == "$":
                end = self._find_closing_math_delimiter(text, i + 1, "$")
                if end != -1:
                    spans.append((i, end + 1))
                    i = end + 1
                    continue

            i += 1

        return spans

    def _overlaps_span_ranges(
        self, start: int, end: int, spans: List[Tuple[int, int]]
    ) -> bool:
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                return True
        return False

    def _handwriting_word_replacements(self, char: str) -> List[str]:
        replacements = HANDWRITING_CONFUSIONS.get(char, [])
        valid: List[str] = []

        for replacement in replacements:
            if replacement.isdigit():
                valid.append(replacement)
            elif replacement.isalpha():
                if char.islower() and replacement.islower():
                    valid.append(replacement)
                elif char.isupper() and replacement.isupper():
                    valid.append(replacement)

        return valid

    def _shift_existing_offsets_after_edit(
        self,
        existing: List[InjectedError],
        edit_start: int,
        original_len: int,
        replacement_len: int,
    ) -> None:
        """Keep previously recorded offsets aligned to the latest corrupted text."""
        delta = replacement_len - original_len
        if delta == 0:
            return

        edit_end_before = edit_start + original_len
        for e in existing:
            if e.char_start >= edit_end_before:
                e.char_start += delta
                e.char_end += delta

    def _inject_spelling_error(
        self, text: str, existing: List[InjectedError]
    ) -> Optional[Tuple[str, InjectedError]]:
        candidates = []
        lower_text = text.lower()
        for word, misspellings in MATH_SPELLING_WORDS.items():
            start = 0
            while True:
                idx = lower_text.find(word, start)
                if idx == -1:
                    break
                if not self._overlaps_existing(idx, idx + len(word), existing):
                    candidates.append((idx, word, misspellings))
                start = idx + 1

        if not candidates:
            return None

        idx, original, misspellings = self.rng.choice(candidates)
        replacement = self.rng.choice(misspellings)

        actual_word = text[idx:idx + len(original)]
        if actual_word[0].isupper():
            replacement = replacement[0].upper() + replacement[1:]

        new_text = text[:idx] + replacement + text[idx + len(original):]
        error = InjectedError(
            error_type="spelling",
            original_span=actual_word,
            corrupted_span=replacement,
            char_start=idx,
            char_end=idx + len(replacement),
        )
        return new_text, error

    def _inject_ocr_symbol_error(
        self, text: str, existing: List[InjectedError]
    ) -> Optional[Tuple[str, InjectedError]]:
        candidates = []
        for symbol, confusions in OCR_SYMBOL_CONFUSIONS.items():
            start = 0
            while True:
                idx = text.find(symbol, start)
                if idx == -1:
                    break
                if not self._overlaps_existing(idx, idx + len(symbol), existing):
                    candidates.append((idx, symbol, confusions))
                start = idx + 1

        if not candidates:
            return None

        idx, original, confusions = self.rng.choice(candidates)
        replacement = self.rng.choice(confusions)
        new_text = text[:idx] + replacement + text[idx + len(original):]
        error = InjectedError(
            error_type="ocr_symbol",
            original_span=original,
            corrupted_span=replacement,
            char_start=idx,
            char_end=idx + len(replacement),
        )
        return new_text, error

    def _inject_ocr_formula_error(
        self, text: str, existing: List[InjectedError]
    ) -> Optional[Tuple[str, InjectedError]]:
        candidates = []
        for pattern, confusions in OCR_FORMULA_CONFUSIONS.items():
            start = 0
            while True:
                idx = text.find(pattern, start)
                if idx == -1:
                    break
                if not self._overlaps_existing(idx, idx + len(pattern), existing):
                    candidates.append((idx, pattern, confusions))
                start = idx + 1

        if not candidates:
            return None

        idx, original, confusions = self.rng.choice(candidates)
        replacement = self.rng.choice(confusions)
        new_text = text[:idx] + replacement + text[idx + len(original):]
        error = InjectedError(
            error_type="ocr_formula",
            original_span=original,
            corrupted_span=replacement,
            char_start=idx,
            char_end=idx + len(replacement),
        )
        return new_text, error

    def _inject_variable_error(
        self, text: str, existing: List[InjectedError]
    ) -> Optional[Tuple[str, InjectedError]]:
        var_pattern = re.compile(r'(?<![a-zA-Z\\])([a-z])(?![a-zA-Z{])')
        matches = list(var_pattern.finditer(text))

        var_positions: Dict[str, List[re.Match]] = {}
        for m in matches:
            var = m.group(1)
            if var in COMMON_VARIABLES:
                var_positions.setdefault(var, []).append(m)

        candidates = [
            (var, positions)
            for var, positions in var_positions.items()
            if len(positions) >= 2
        ]

        if not candidates:
            return None

        var, positions = self.rng.choice(candidates)
        target_match = self.rng.choice(positions[1:])
        idx = target_match.start()

        if self._overlaps_existing(idx, idx + 1, existing):
            return None

        other_vars = [v for v in COMMON_VARIABLES if v != var]
        replacement = self.rng.choice(other_vars)

        new_text = text[:idx] + replacement + text[idx + 1:]
        error = InjectedError(
            error_type="variable",
            original_span=var,
            corrupted_span=replacement,
            char_start=idx,
            char_end=idx + 1,
        )
        return new_text, error

    def _inject_handwriting_confusion_error(
        self, text: str, existing: List[InjectedError]
    ) -> Optional[Tuple[str, InjectedError]]:
        math_spans = self._find_latex_math_spans(text)
        candidates = []
        for match in HANDWRITING_WORD_PATTERN.finditer(text):
            word_start, word_end = match.span()
            if self._overlaps_existing(word_start, word_end, existing):
                continue
            if self._overlaps_span_ranges(word_start, word_end, math_spans):
                continue

            word = match.group(0)
            for offset in range(1, len(word) - 1):
                replacements = self._handwriting_word_replacements(word[offset])
                if not replacements:
                    continue
                candidates.append((word_start, word_end, word, offset, replacements))

        if not candidates:
            return None

        word_start, word_end, original_word, offset, replacements = self.rng.choice(candidates)
        replacement = self.rng.choice(replacements)
        corrupted_word = (
            original_word[:offset] + replacement + original_word[offset + 1:]
        )
        new_text = text[:word_start] + corrupted_word + text[word_end:]
        error = InjectedError(
            error_type="handwriting_confusion",
            original_span=original_word,
            corrupted_span=corrupted_word,
            char_start=word_start,
            char_end=word_start + len(corrupted_word),
        )
        return new_text, error
