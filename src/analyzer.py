import os
import torch
import base64
import argparse
from tqdm import tqdm

# Optional imports for local models
try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
    from transformers.image_utils import load_image
except ImportError:
    pass # Managed if local not used

# Optional import for OpenAI
try:
    from openai import OpenAI
except ImportError:
    pass

class PaperContent:
    def __init__(self, paper_dir):
        self.paper_dir = paper_dir
        self.pages = []
        self._load_pages()

    def _load_pages(self):
        pages_dir = os.path.join(self.paper_dir, "pages")
        if not os.path.exists(pages_dir):
            print(f"Warning: Pages directory not found at {pages_dir}")
            return
            
        # listing page_0, page_1...
        page_folders = sorted([d for d in os.listdir(pages_dir) if d.startswith("page_") and os.path.isdir(os.path.join(pages_dir, d))], 
                              key=lambda x: int(x.split('_')[1]))
        
        for folder in page_folders:
            folder_path = os.path.join(pages_dir, folder)
            page_num = folder.split('_')[1]
            
            # Load Text
            text_path = os.path.join(folder_path, "result.mmd")
            text_content = ""
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            
            # Load Pages
            # We prioritize cropped images if they exist, but fallback to original.jpg
            # deepseek-ocr didn't find crops in our test, so we need original.jpg as fallback.
            
            images_dir = os.path.join(folder_path, "images")
            images_to_load = []
            
            # Check for crops
            if os.path.exists(images_dir):
                 for img_file in sorted(os.listdir(images_dir)):
                     if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                         images_to_load.append(os.path.join(images_dir, img_file))
            
            # Fallback to original if no crops
            if not images_to_load:
                orig_path = os.path.join(folder_path, "original.jpg")
                if os.path.exists(orig_path):
                    images_to_load.append(orig_path)
            
            self.pages.append({
                "page_num": page_num,
                "text": text_content,
                "images": images_to_load
            })

class BaseAnalyzer:
    def analyze(self, paper_content):
        raise NotImplementedError

    def _get_system_prompt(self):
        return """You are an expert Computer Science researcher and reviewer. 
You are analyzing a NeurIPS paper. The input consists of text extracted from each page, followed immediately by any figure/table images found on that page.
Use BOTH the text and the visual information to analyze the paper.

Report Structure:
1. **Background**: Problem context.
2. **Research Gap**: What is missing in literature?
3. **Method**: Technical approach (reference specific figures if relevant).
4. **Dataset**: Datasets used.
5. **Evaluation**: Metrics and baselines.
6. **Critical Thinking**: Strengths, weaknesses, and novelty judgment.
"""

class OpenAIAnalyzer(BaseAnalyzer):
    def __init__(self, model_name="gpt-5", api_key=None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key) 

    def analyze(self, paper_content):
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": []}
        ]
        
        user_content = messages[1]["content"]
        user_content.append({"type": "text", "text": "Please analyze this paper based on the following pages:\n\n"})
        
        for page in paper_content.pages:
            # Add Text
            user_content.append({
                "type": "text", 
                "text": f"--- Page {page['page_num']} Text ---\n{page['text']}\n"
            })
            
            # Add Images (if any)
            if page['images']:
                user_content.append({"type": "text", "text": f"\n[Figures/Tables found on Page {page['page_num']}]:\n"})
                for img_path in page['images']:
                     if os.path.exists(img_path):
                        base64_image = self._encode_image(img_path)
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" 
                            }
                        })
                user_content.append({"type": "text", "text": "\n"})
                
            user_content.append({"type": "text", "text": f"\n[End of Page {page['page_num']}]\n\n"})

        print(f"Sending request to OpenAI ({self.model_name})...")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI: {e}"

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class LocalVLMAnalyzer(BaseAnalyzer):
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Thinking", device='cuda'):
        self.model_name = model_name
        self.device = device
        self._load_model()
        
    def _load_model(self):
        print(f"Loading local VLM: {self.model_name}...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, 
                device_map="auto", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            ).eval()
            print("Model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def analyze(self, paper_content):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._get_system_prompt() + "\n\n"}
                ]
            }
        ]
        
        current_content = conversation[0]["content"]
        
        images = []
        for page in paper_content.pages:
            current_content.append({"type": "text", "text": f"\n--- Page {page['page_num']} ---\n{page['text']}\n"})
            
            if page['images']:
                current_content.append({"type": "text", "text": f"\n[Figures for Page {page['page_num']}]:\n"})
                for img_path in page['images']:
                    img = load_image(img_path)
                    
                    # # Resize to avoid OOM
                    # if max(img.size) > 640:
                    #     img.thumbnail((640, 640))
                        
                    images.append(img)
                    current_content.append({"type": "image", "image": img})
                current_content.append({"type": "text", "text": "\n"})

        # Prepare for inference
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Processor handling of images depends on the specific model's chat template implementation.
        # If apply_chat_template handles images (by replacing placeholders), we pass images to the processor.
        
        inputs = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        print("Generating local VLM analysis...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper_folder", help="Path to the paper extracted folder")
    parser.add_argument("--provider", choices=["local", "online"], default="local", help="Analysis provider")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--api_key", default=None, help="OpenAI API Key (optional if in env)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.paper_folder):
        print(f"Error: {args.paper_folder} does not exist.")
        return

    content = PaperContent(args.paper_folder)
    
    analyzer = None
    if args.provider == "online":
         model = args.model if args.model else "gpt-5" # Default to user request
         analyzer = OpenAIAnalyzer(model_name=model, api_key=args.api_key)
    else:
         model = args.model if args.model else "Qwen/Qwen3-VL-4B-Thinking"
         analyzer = LocalVLMAnalyzer(model_name=model)
         
    print(f"Starting analysis with {args.provider} model: {model}...")
    report = analyzer.analyze(content)
    
    report_path = os.path.join(args.paper_folder, "analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(f"Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
