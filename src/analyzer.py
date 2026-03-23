import argparse
import base64
import mimetypes
import os

import torch

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )
    from transformers.image_utils import load_image
except ImportError:
    AutoConfig = None
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None
    load_image = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LOCAL_OPENAI_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_LOCAL_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_TEXT_ANALYSIS_MAX_INPUT_TOKENS = 6000
LOCAL_TEXT_ANALYSIS_MAX_PAGE_TOKENS = 4500
PROVIDER_ALIASES = {"online": "openai", "local_openai": "local-openai"}


class PaperContent:
    def __init__(self, paper_dir):
        self.paper_dir = os.fspath(paper_dir)
        self.pages = []
        self._load_pages()

    def _load_pages(self):
        pages_dir = os.path.join(self.paper_dir, "pages")
        if not os.path.exists(pages_dir):
            print(f"Warning: Pages directory not found at {pages_dir}")
            return

        page_folders = sorted(
            [
                folder
                for folder in os.listdir(pages_dir)
                if folder.startswith("page_")
                and os.path.isdir(os.path.join(pages_dir, folder))
            ],
            key=lambda folder: int(folder.split("_")[1]),
        )

        for folder in page_folders:
            folder_path = os.path.join(pages_dir, folder)
            page_num = folder.split("_")[1]

            text_path = os.path.join(folder_path, "result.mmd")
            text_content = ""
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as file_handle:
                    text_content = file_handle.read()

            images_dir = os.path.join(folder_path, "images")
            images_to_load = []

            if os.path.exists(images_dir):
                for img_file in sorted(os.listdir(images_dir)):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        images_to_load.append(os.path.join(images_dir, img_file))

            if not images_to_load:
                orig_path = os.path.join(folder_path, "original.jpg")
                if os.path.exists(orig_path):
                    images_to_load.append(orig_path)

            self.pages.append(
                {
                    "page_num": page_num,
                    "text": text_content,
                    "images": images_to_load,
                }
            )


class BaseAnalyzer:
    def analyze(self, paper_content):
        raise NotImplementedError

    def _get_system_prompt(self):
        return """You are an expert Computer Science researcher and reviewer.
You are analyzing a NeurIPS paper. The input consists of OCR text extracted from each page and, when supported by the model, figure or table images found on that page.
Use the available information to analyze the paper.

Report Structure:
1. **Background**: Problem context.
2. **Research Gap**: What is missing in literature?
3. **Method**: Technical approach (reference specific figures if relevant).
4. **Dataset**: Datasets used.
5. **Evaluation**: Metrics and baselines.
6. **Critical Thinking**: Strengths, weaknesses, and novelty judgment.
"""

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _guess_media_type(self, image_path):
        media_type, _ = mimetypes.guess_type(image_path)
        return media_type or "image/jpeg"


class OpenAIAnalyzer(BaseAnalyzer):
    def __init__(self, model_name=DEFAULT_OPENAI_MODEL, api_key=None, base_url=None):
        if OpenAI is None:
            raise ImportError("OpenAI SDK is not installed. Install the 'openai' package.")

        self.model_name = model_name
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()

    def analyze(self, paper_content):
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": []},
        ]

        user_content = messages[1]["content"]
        user_content.append(
            {
                "type": "text",
                "text": "Please analyze this paper based on the following pages:\n\n",
            }
        )

        for page in paper_content.pages:
            user_content.append(
                {
                    "type": "text",
                    "text": f"--- Page {page['page_num']} Text ---\n{page['text']}\n",
                }
            )

            if page["images"]:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"\n[Figures/Tables found on Page {page['page_num']}]:\n",
                    }
                )
                for img_path in page["images"]:
                    if os.path.exists(img_path):
                        base64_image = self._encode_image(img_path)
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{self._guess_media_type(img_path)};"
                                        f"base64,{base64_image}"
                                    ),
                                    "detail": "high",
                                },
                            }
                        )
                user_content.append({"type": "text", "text": "\n"})

            user_content.append(
                {"type": "text", "text": f"\n[End of Page {page['page_num']}]\n\n"}
            )

        print(f"Sending request to OpenAI ({self.model_name})...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2048,
        )
        return response.choices[0].message.content


class LocalOpenAIAnalyzer(BaseAnalyzer):
    def __init__(self, model_name=DEFAULT_LOCAL_OPENAI_MODEL, api_key=None, base_url=None):
        if OpenAI is None:
            raise ImportError("OpenAI SDK is not installed. Install the 'openai' package.")

        self.model_name = model_name
        self.base_url = base_url or os.getenv("LOCAL_OPENAI_BASE_URL") or DEFAULT_LOCAL_OPENAI_BASE_URL
        resolved_api_key = api_key or os.getenv("LOCAL_OPENAI_API_KEY") or "EMPTY"
        self.client = OpenAI(api_key=resolved_api_key, base_url=self.base_url)

    def analyze(self, paper_content):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._get_system_prompt()
                        + "\n\nPlease analyze this paper based on the OCR text and page images below.\n",
                    }
                ],
            }
        ]

        user_content = messages[0]["content"]
        for page in paper_content.pages:
            user_content.append(
                {
                    "type": "text",
                    "text": f"\n--- Page {page['page_num']} Text ---\n{page['text']}\n",
                }
            )

            for img_path in page["images"]:
                if not os.path.exists(img_path):
                    continue

                base64_image = self._encode_image(img_path)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{self._guess_media_type(img_path)};"
                                f"base64,{base64_image}"
                            )
                        },
                    }
                )

        print(f"Sending request to local OpenAI-compatible endpoint ({self.model_name})...")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=32768,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={"top_k": 20},
        )
        return response.choices[0].message.content


class ClaudeAnalyzer(BaseAnalyzer):
    def __init__(self, model_name=DEFAULT_CLAUDE_MODEL, api_key=None):
        if Anthropic is None:
            raise ImportError("Anthropic SDK is not installed. Install the 'anthropic' package.")

        self.model_name = model_name
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def analyze(self, paper_content):
        user_content = [
            {
                "type": "text",
                "text": "Please analyze this paper based on the following OCR pages and extracted figures/tables.\n",
            }
        ]

        for page in paper_content.pages:
            user_content.append(
                {"type": "text", "text": f"\n--- Page {page['page_num']} Figures/Tables ---\n"}
            )

            for img_path in page["images"]:
                if os.path.exists(img_path):
                    user_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": self._guess_media_type(img_path),
                                "data": self._encode_image(img_path),
                            },
                        }
                    )

            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"\n--- Page {page['page_num']} Text ---\n"
                        f"{page['text']}\n"
                        f"[End of Page {page['page_num']}]\n"
                    ),
                }
            )

        print(f"Sending request to Claude ({self.model_name})...")
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            system=self._get_system_prompt(),
            messages=[{"role": "user", "content": user_content}],
        )
        return _extract_anthropic_text(response)


class LocalVLMAnalyzer(BaseAnalyzer):
    def __init__(self, model_name=DEFAULT_LOCAL_MODEL):
        if AutoConfig is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "Transformers local-model dependencies are not installed. "
                "Install the packages in requirements.txt."
            )

        self.model_name = model_name
        self.config = None
        self.model = None
        self.model_mode = None
        self.processor = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading local model: {self.model_name}...")
        try:
            self.config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except ValueError as exc:
            if "qwen3_5" in str(exc):
                raise ValueError(
                    "Qwen/Qwen3.5-2B is not supported by transformers==4.46.3. "
                    "Use the default local model Qwen/Qwen2.5-1.5B-Instruct, "
                    "or serve Qwen/Qwen3.5-2B behind an OpenAI-compatible endpoint "
                    "and run with --provider local-openai."
                ) from exc
            raise

        if self._should_use_multimodal_mode():
            self._load_multimodal_model()
        else:
            self._load_text_model()

    def _preferred_torch_dtype(self):
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _should_use_multimodal_mode(self):
        if AutoModelForImageTextToText is None or AutoProcessor is None or load_image is None:
            return False

        model_name = self.model_name.lower()
        return any(token in model_name for token in ("-vl", "/vl", "vision", "omni"))

    def _load_text_model(self):
        self.model_mode = "text"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self._preferred_torch_dtype(),
        ).eval()
        print("Loaded local text model.")

    def _load_multimodal_model(self):
        self.model_mode = "multimodal"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=self._preferred_torch_dtype(),
        ).eval()
        print("Loaded local multimodal model.")

    def analyze(self, paper_content):
        if self.model_mode == "multimodal":
            return self._analyze_multimodal(paper_content)
        return self._analyze_text_only(paper_content)

    def _analyze_multimodal(self, paper_content):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self._get_system_prompt() + "\n\n"}],
            }
        ]

        current_content = conversation[0]["content"]
        images = []

        for page in paper_content.pages:
            current_content.append(
                {
                    "type": "text",
                    "text": f"\n--- Page {page['page_num']} ---\n{page['text']}\n",
                }
            )

            if page["images"]:
                current_content.append(
                    {"type": "text", "text": f"\n[Figures for Page {page['page_num']}]:\n"}
                )
                for img_path in page["images"]:
                    img = load_image(img_path)
                    images.append(img)
                    current_content.append({"type": "image", "image": img})
                current_content.append({"type": "text", "text": "\n"})

        text_prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        print("Generating local multimodal analysis...")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _analyze_text_only(self, paper_content):
        summary_pages = self._prepare_summary_pages(paper_content.pages)
        summary_content = type("SummaryPaperContent", (), {"pages": summary_pages})()
        prompt = self._build_chat_prompt(self._build_text_only_prompt(summary_content))
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        if prompt_tokens <= LOCAL_TEXT_ANALYSIS_MAX_INPUT_TOKENS:
            print("Generating local text-only analysis...")
            return self._generate_text_response(prompt)

        print("Paper is too large for a single local prompt. Running chunked analysis...")
        chunk_summaries = []
        for chunk_index, page_chunk in enumerate(self._chunk_pages(summary_pages), start=1):
            chunk_prompt = self._build_chat_prompt(
                self._build_chunk_prompt(page_chunk, chunk_index)
            )
            chunk_summary = self._generate_text_response(chunk_prompt, max_new_tokens=768)
            chunk_summaries.append(
                f"## Chunk {chunk_index}\n"
                f"Pages: {page_chunk[0]['page_num']} to {page_chunk[-1]['page_num']}\n\n"
                f"{chunk_summary}"
            )

        synthesis_prompt = self._build_chat_prompt(
            self._build_synthesis_prompt(chunk_summaries)
        )
        print("Generating final synthesized report from chunk summaries...")
        return self._generate_text_response(synthesis_prompt, max_new_tokens=1024)

    def _build_text_only_prompt(self, paper_content):
        parts = [
            "Please analyze this paper from the OCR output below.",
            "If image evidence is unavailable to the model, rely on the OCR text and note uncertainty where needed.",
        ]

        for page in paper_content.pages:
            parts.append(f"\n--- Page {page['page_num']} ---")
            if page["images"]:
                parts.append(f"[Extracted images on this page: {len(page['images'])}]")
            parts.append(page["text"])

        return "\n".join(parts)

    def _build_chat_prompt(self, user_content):
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            self._get_system_prompt()
            + "\n\n"
            + user_content
            + "\n\nPlease generate the analysis report."
        )

    def _generate_text_response(self, prompt, max_new_tokens=1024):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(
            generated_ids_trimmed[0],
            skip_special_tokens=True,
        ).strip()

    def _chunk_pages(self, pages):
        current_chunk = []
        for page in pages:
            candidate_chunk = current_chunk + [page]
            candidate_prompt = self._build_chat_prompt(
                self._build_chunk_prompt(candidate_chunk, len(current_chunk) + 1)
            )
            candidate_tokens = self.tokenizer(
                candidate_prompt,
                return_tensors="pt",
            ).input_ids.shape[1]

            if current_chunk and candidate_tokens > LOCAL_TEXT_ANALYSIS_MAX_INPUT_TOKENS:
                yield current_chunk
                current_chunk = [page]
            else:
                current_chunk = candidate_chunk

        if current_chunk:
            yield current_chunk

    def _prepare_summary_pages(self, pages):
        selected_pages = self._select_summary_pages(pages)
        prepared_pages = []
        for page in selected_pages:
            prepared_page = dict(page)
            prepared_page["text"] = self._truncate_page_text(page["text"])
            prepared_pages.append(prepared_page)
        return prepared_pages

    def _select_summary_pages(self, pages):
        stop_markers = (
            "## references",
            "## appendix",
            "## neurips paper checklist",
        )
        selected_pages = []

        for page in pages:
            normalized_text = page["text"].lower()
            if any(marker in normalized_text for marker in stop_markers):
                break
            selected_pages.append(page)

        return selected_pages or pages

    def _truncate_page_text(self, text):
        token_count = self.tokenizer(text, return_tensors="pt").input_ids.shape[1]
        if token_count <= LOCAL_TEXT_ANALYSIS_MAX_PAGE_TOKENS:
            return text

        encoded = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        truncated_ids = encoded[:LOCAL_TEXT_ANALYSIS_MAX_PAGE_TOKENS]
        truncated_text = self.tokenizer.decode(truncated_ids, skip_special_tokens=True)
        return (
            truncated_text
            + "\n\n[Truncated for local analysis because this page exceeded the token budget.]"
        )

    def _build_chunk_prompt(self, page_chunk, chunk_index):
        parts = [
            f"You are analyzing chunk {chunk_index} of a paper.",
            "Produce structured notes under these headings only:",
            "Background",
            "Research Gap",
            "Method",
            "Dataset",
            "Evaluation",
            "Critical Thinking",
            "If a heading is unsupported by this chunk, explicitly say that the evidence is not present in this chunk.",
        ]

        for page in page_chunk:
            parts.append(f"\n--- Page {page['page_num']} ---")
            parts.append(page["text"])

        return "\n".join(parts)

    def _build_synthesis_prompt(self, chunk_summaries):
        parts = [
            "The following are chunk-level analysis notes for one paper.",
            "Merge them into one final report using this structure exactly:",
            "1. Background",
            "2. Research Gap",
            "3. Method",
            "4. Dataset",
            "5. Evaluation",
            "6. Critical Thinking",
            "Prefer the most specific evidence. If chunk notes conflict, mention the uncertainty briefly.",
            "",
            "\n\n".join(chunk_summaries),
        ]
        return "\n".join(parts)


def normalize_provider(provider):
    return PROVIDER_ALIASES.get(provider, provider)


def _resolve_api_key(provider, api_key=None):
    if api_key:
        return api_key

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")

    if provider == "claude":
        return os.getenv("CLAUDE_CODE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

    if provider == "local-openai":
        return os.getenv("LOCAL_OPENAI_API_KEY") or "EMPTY"

    return None


def build_analyzer(provider="local", model_name=None, api_key=None, base_url=None):
    provider = normalize_provider(provider)

    if provider == "openai":
        resolved_key = _resolve_api_key(provider, api_key)
        return OpenAIAnalyzer(
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            api_key=resolved_key,
            base_url=base_url,
        )

    if provider == "claude":
        resolved_key = _resolve_api_key(provider, api_key)
        return ClaudeAnalyzer(model_name=model_name or DEFAULT_CLAUDE_MODEL, api_key=resolved_key)

    if provider == "local-openai":
        resolved_key = _resolve_api_key(provider, api_key)
        return LocalOpenAIAnalyzer(
            model_name=model_name or DEFAULT_LOCAL_OPENAI_MODEL,
            api_key=resolved_key,
            base_url=base_url,
        )

    if provider == "local":
        return LocalVLMAnalyzer(model_name=model_name or DEFAULT_LOCAL_MODEL)

    raise ValueError(f"Unsupported provider: {provider}")


def analyze_paper_folder(paper_folder, analyzer, report_filename="analysis_report.md"):
    paper_folder = os.fspath(paper_folder)
    content = PaperContent(paper_folder)
    report = analyzer.analyze(content)

    report_path = os.path.join(paper_folder, report_filename)
    with open(report_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(report)

    print(f"Analysis saved to {report_path}")
    return report_path


def _extract_anthropic_text(response):
    chunks = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")

        if block_type != "text":
            continue

        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")

        if text:
            chunks.append(text)

    return "\n".join(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_folder", help="Path to the paper extracted folder")
    parser.add_argument(
        "--provider",
        choices=["local", "local-openai", "openai", "claude", "online"],
        default="local",
        help="Analysis provider",
    )
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api_key", default=None, help="API key for online providers")
    parser.add_argument("--base_url", default=None, help="Base URL for OpenAI-compatible providers")

    args = parser.parse_args()

    if not os.path.exists(args.paper_folder):
        print(f"Error: {args.paper_folder} does not exist.")
        return

    analyzer = build_analyzer(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    print(f"Starting analysis with {normalize_provider(args.provider)} model...")
    analyze_paper_folder(args.paper_folder, analyzer)


if __name__ == "__main__":
    main()
