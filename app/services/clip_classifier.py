import os
import io
from typing import List, Dict, Any, Optional, Tuple
import torch
from PIL import Image
import clip


class ClipClassifier:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        self._text_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def load_image(self, data: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def encode_images(self, images: List[Image.Image], batch_size: int = 16) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_t = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_t)
            tensors.append(feats)
        out = torch.cat(tensors, dim=0)
        out = out / out.norm(dim=-1, keepdim=True)
        return out

    def build_concepts(self, context: Optional[str] = None) -> List[str]:
        base = [
            "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
            "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
            "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
            "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
            "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
            "hair drier","toothbrush"
        ]
        if context:
            ctx = [f"a photo of a {w}" for w in base] + [f"a {w}" for w in base]
            return ctx
        return [f"a photo of a {w}" for w in base]

    def encode_texts(self, prompts: List[str]) -> torch.Tensor:
        key = tuple(prompts)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            text_feats = self.model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        self._text_cache[key] = text_feats
        return text_feats

    def classify(self, image_feats: torch.Tensor, text_feats: torch.Tensor, labels: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        sims = image_feats @ text_feats.T
        probs = sims.softmax(dim=-1)
        topk = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
        results: List[Dict[str, Any]] = []
        for i in range(probs.shape[0]):
            idxs = topk.indices[i].tolist()
            scs = topk.values[i].tolist()
            items = [{"label": labels[j], "confidence": float(scs[k])} for k, j in enumerate(idxs)]
            results.append({"top1": items[0]["label"], "labels": items, "max_confidence": float(scs[0])})
        return results

    def filter_below_threshold(self, items: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        return [x for x in items if x["max_confidence"] >= threshold]

    def group_by_top1(self, items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            k = it["top1"]
            groups.setdefault(k, []).append(it)
        return groups

    def export_jsonl(self, items: List[Dict[str, Any]], paths: List[str], out_path: str) -> str:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, it in enumerate(items):
                rec = {"path": paths[i], "top1": it["top1"], "labels": it["labels"], "confidence": it["max_confidence"]}
                f.write(f"{rec}\n")
        return out_path