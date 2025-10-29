# image_prompt_reverse.py
import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import folder_paths

# BLIP model for image captioning
from transformers import BlipProcessor, BlipForConditionalGeneration

# CLIP model for tag prediction
import clip

class ImagePromptReverse:
    """
    ComfyUI Custom Node for Reverse Image Prompt Generation
    ComfyUI自定义反推图片提示词节点
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = None
        self.blip_model = None
        self.clip_model = None
        self.clip_preprocess = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_type": (["blip", "clip", "ensemble"], {
                    "default": "ensemble"
                }),
                "prompt_style": (["detailed", "simple", "artistic", "photographic"], {
                    "default": "detailed"
                }),
                "language": (["english", "chinese", "bilingual"], {
                    "default": "bilingual"
                }),
                "max_length": ("INT", {
                    "default": 30, 
                    "min": 10, 
                    "max": 100,
                    "step": 5
                }),
            },
            "optional": {
                "custom_prompt_template": ("STRING", {
                    "multiline": True,
                    "default": "A {style} image of"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt_en", "prompt_cn")
    FUNCTION = "reverse_prompt"
    CATEGORY = "image/prompt"
    
    def load_models(self, model_type):
        """Load required models 加载所需模型"""
        try:
            if model_type in ["blip", "ensemble"] and self.blip_model is None:
                print("Loading BLIP model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            
            if model_type in ["clip", "ensemble"] and self.clip_model is None:
                print("Loading CLIP model...")
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def preprocess_image(self, image):
        """Convert ComfyUI image to PIL format 将ComfyUI图像转换为PIL格式"""
        # ComfyUI image tensor: [batch, height, width, channels]
        image = image[0]  # Take first image in batch
        image = image.cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)
    
    def blip_caption(self, image, max_length):
        """Generate caption using BLIP 使用BLIP生成描述"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            out = self.blip_model.generate(**inputs, max_length=max_length, num_beams=5)
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"BLIP caption error: {e}")
            return ""
    
    def clip_tags(self, image, top_k=10):
        """Generate tags using CLIP 使用CLIP生成标签"""
        try:
            # Common tags for classification
            tags = [
                "photo", "painting", "drawing", "digital art", "landscape", "portrait", 
                "abstract", "realistic", "surreal", "minimalist", "vibrant colors",
                "dark", "bright", "detailed", "simple", "professional", "amateur",
                "nature", "city", "indoor", "outdoor", "people", "animals", "objects",
                "fantasy", "sci-fi", "historical", "modern", "vintage", "futuristic"
            ]
            
            preprocessed_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = torch.cat([clip.tokenize(f"a photo of {tag}") for tag in tags]).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(preprocessed_image)
                text_features = self.clip_model.encode_text(text_inputs)
                
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(top_k)
            
            top_tags = [tags[idx] for idx in indices.cpu().numpy()]
            return ", ".join(top_tags)
            
        except Exception as e:
            print(f"CLIP tags error: {e}")
            return ""
    
    def translate_to_chinese(self, english_text):
        """Simple translation to Chinese 简单翻译成中文"""
        # This is a basic translation mapping - in practice you might want to use a proper translation API
        translation_map = {
            "photo": "照片",
            "painting": "绘画",
            "drawing": "素描",
            "digital art": "数字艺术",
            "landscape": "风景",
            "portrait": "肖像",
            "abstract": "抽象",
            "realistic": "写实",
            "surreal": "超现实",
            "minimalist": "极简主义",
            "vibrant colors": "鲜艳色彩",
            "dark": "暗调",
            "bright": "明亮",
            "detailed": "细节丰富",
            "simple": "简单",
            "professional": "专业",
            "amateur": "业余",
            "nature": "自然",
            "city": "城市",
            "indoor": "室内",
            "outdoor": "室外",
            "people": "人物",
            "animals": "动物",
            "objects": "物体",
            "fantasy": "奇幻",
            "sci-fi": "科幻",
            "historical": "历史",
            "modern": "现代",
            "vintage": "复古",
            "futuristic": "未来主义",
            "a": "一张",
            "of": "的",
            "with": "带有",
            "and": "和",
            "in": "在",
            "on": "在",
            "at": "在"
        }
        
        chinese_text = english_text
        for eng, chi in translation_map.items():
            chinese_text = chinese_text.replace(eng, chi)
        
        return chinese_text
    
    def apply_prompt_style(self, prompt, style):
        """Apply different prompt styles 应用不同的提示词风格"""
        if style == "detailed":
            return f"highly detailed, professional, 4K resolution, {prompt}"
        elif style == "simple":
            return prompt
        elif style == "artistic":
            return f"artistic style, creative composition, {prompt}"
        elif style == "photographic":
            return f"photorealistic, professional photography, sharp focus, {prompt}"
        else:
            return prompt
    
    def reverse_prompt(self, image, model_type, prompt_style, language, max_length, custom_prompt_template=None):
        """Main function to generate reverse prompts 生成反推提示词的主函数"""
        
        # Load models if needed
        self.load_models(model_type)
        
        # Convert ComfyUI image to PIL
        pil_image = self.preprocess_image(image)
        
        # Generate prompts based on model type
        prompt_en = ""
        
        if model_type == "blip" or model_type == "ensemble":
            blip_caption = self.blip_caption(pil_image, max_length)
            if blip_caption:
                prompt_en = blip_caption
        
        if model_type == "clip" or (model_type == "ensemble" and not prompt_en):
            clip_tags = self.clip_tags(pil_image)
            if clip_tags:
                prompt_en = f"a photo featuring {clip_tags}"
        
        # Apply prompt style
        if prompt_en:
            prompt_en = self.apply_prompt_style(prompt_en, prompt_style)
            
            # Apply custom template if provided
            if custom_prompt_template and "{style}" in custom_prompt_template:
                prompt_en = custom_prompt_template.replace("{style}", prompt_style) + " " + prompt_en
        
        # Generate Chinese prompt
        prompt_cn = self.translate_to_chinese(prompt_en) if prompt_en else ""
        
        # Handle language selection
        if language == "english":
            prompt_cn = ""
        elif language == "chinese":
            prompt_en, prompt_cn = prompt_cn, prompt_en
        
        print(f"Generated English prompt: {prompt_en}")
        print(f"Generated Chinese prompt: {prompt_cn}")
        
        return (prompt_en, prompt_cn)

# Register the node
NODE_CLASS_MAPPINGS = {
    "ImagePromptReverse": ImagePromptReverse
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePromptReverse": "Image Prompt Reverse (图片反推提示词)"
}