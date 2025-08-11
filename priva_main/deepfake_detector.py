import io, math
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as F
import timm

from facenet_pytorch import MTCNN
from priva_main.config import MODELS_DIR

# ---------- helpers ----------
def _ensure_rgb_np(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image

def _to_pil(image):
    return image if isinstance(image, Image.Image) else Image.fromarray(image)

def _resize_max(img, max_side=720):
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def _affine_align(rgb: np.ndarray, lm: np.ndarray, out_size: int = 256) -> np.ndarray:
    """
    Align face using 5-point landmarks (from MTCNN) to a canonical template.
    lm: shape (5,2) order: [left_eye, right_eye, nose, mouth_left, mouth_right]
    """
    # canonical points (rough, from InsightFace template scaled to out_size)
    t = np.array([
        [0.341, 0.412],  # left eye
        [0.655, 0.412],  # right eye
        [0.498, 0.537],  # nose
        [0.369, 0.704],  # mouth left
        [0.626, 0.704],  # mouth right
    ], dtype=np.float32)
    dst = (t * out_size).astype(np.float32)
    src = lm.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
    if M is None:
        return cv2.resize(rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.warpAffine(rgb, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# ---------- learned models ----------
class _TimmBinary(nn.Module):
    def __init__(self, backbone: str, num_classes: int = 1):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=False, num_classes=num_classes, in_chans=3, global_pool='avg')
    def forward(self, x):
        return self.model(x)

def _load_sd(model, path: str) -> bool:
    try:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            # strip common prefixes
            fixed = {}
            for k, v in sd.items():
                k = k.replace("module.", "")
                k = k.replace("model.", "")
                fixed[k] = v
            sd = fixed
        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False

# ---------- Detector ----------
class DeepfakeDetector:
    """
    Two-path detector:
      1) Learned ensemble (EfficientNet/Xception) with face alignment + 10-crop TTA (if weights available).
      2) Strong artifact model (no weights) as fallback; also blended with learned logits for robustness.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # MTCNN face detector with landmarks
        self.face_detector = MTCNN(margin=20, select_largest=True, post_process=False, device=self.device, keep_all=True)
        # try to load models
        self.ensemble = self._build_ensemble()
        self.has_learned = len(self.ensemble) > 0

    # ---------- ensemble build ----------
    def _build_ensemble(self):
        models = []

        # EffNet-B4
        effb4 = MODELS_DIR / "dfdc_effb4.pth"
        if effb4.exists():
            m = _TimmBinary("tf_efficientnet_b4_ns"); ok = _load_sd(m, str(effb4))
            if ok: models.append(("EffB4", m.to(self.device).eval(), 380))

        # Xception
        xcp = MODELS_DIR / "ffpp_xception_c23.pth"
        if xcp.exists():
            # timm xception: input 299
            m = _TimmBinary("xception"); ok = _load_sd(m, str(xcp))
            if ok: models.append(("Xception", m.to(self.device).eval(), 299))

        return models

    # ---------- TTA ----------
    def _make_tta_batch(self, pil: Image.Image, size: int):
        bigger = transforms.Resize(size + 32)(pil)
        crops = list(transforms.FiveCrop(size)(bigger))
        flipped = F.hflip(bigger)
        crops += list(transforms.FiveCrop(size)(flipped))
        norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        tens = torch.stack([norm(transforms.ToTensor()(c)) for c in crops], dim=0)  # [10,3,H,W]
        return tens

    @torch.no_grad()
    def _predict_model_prob(self, model: nn.Module, pil: Image.Image, size: int) -> float:
        x = self._make_tta_batch(pil, size).to(self.device)  # [10,3,H,W]
        logits = model(x).float().view(-1)                    # [10]
        probs = torch.sigmoid(logits).mean().item()
        return float(np.clip(probs, 0, 1))

    # ---------- artifacts (fallback/aux) ----------
    def _artifact_prob(self, pil_img: Image.Image) -> Tuple[float, Dict[str, float]]:
        import numpy as np
        p = np.array(pil_img.convert("RGB"))
        p = _resize_max(p, 512)
        gray = cv2.cvtColor(p, cv2.COLOR_RGB2GRAY)

        # ELA
        b = io.BytesIO(); pil_img.save(b, 'JPEG', quality=90); b.seek(0)
        comp = Image.open(b).convert('RGB')
        ela = float(np.clip(np.mean(np.abs(np.asarray(pil_img, dtype=np.int16) - np.asarray(comp, dtype=np.int16))) / 16.0, 0, 1))

        # High-frequency energy ratio
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        f = np.fft.fft2(gray); fshift = np.fft.fftshift(f); mag = np.log(np.abs(fshift) + 1e-6)
        h,w = mag.shape; r = int(min(h,w)*0.12)
        c = mag[h//2-r:h//2+r, w//2-r:w//2+r]; o = mag.copy(); o[h//2-r:h//2+r, w//2-r:w//2+r] = 0
        ratio = np.mean(o)/(np.mean(c)+1e-6)
        hf = float(np.clip(0.5*(1 - np.exp(-lap/250.0)) + 0.5*np.tanh(ratio-1.0), 0, 1))

        # Blockiness
        v = np.mean(np.abs(np.diff(gray[:, ::8], axis=1)))/255.0
        h_ = np.mean(np.abs(np.diff(gray[::8, :], axis=0)))/255.0
        block = float(np.clip(v + h_, 0, 1))

        # Color-noise correlation
        blur = cv2.medianBlur(p,5); noise = p.astype(np.float32) - blur.astype(np.float32)
        r,g,b_ = [noise[...,i].reshape(-1) for i in range(3)]
        if min(np.std(r),np.std(g),np.std(b_)) < 1e-6:
            color = 0.0
        else:
            rg = np.corrcoef(r,g)[0,1]; gb = np.corrcoef(g,b_)[0,1]; rb = np.corrcoef(r,b_)[0,1]
            mc = np.clip((rg+gb+rb)/3.0, -1, 1)
            color = float(np.clip(1.0 - (mc+1)/2.0, 0, 1))

        # LBP entropy
        small = cv2.resize(gray,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        c_ = small[1:-1,1:-1]
        nbs = [small[1:-1,2:], small[2:,2:], small[2:,1:-1], small[2:,:-2],
               small[1:-1,:-2], small[:-2,:-2], small[:-2,1:-1], small[:-2,2:]]
        codes = np.zeros_like(c_, dtype=np.uint8)
        for i,nb in enumerate(nbs):
            codes |= ((nb >= c_) << i).astype(np.uint8)
        hist,_ = np.histogram(codes, bins=256, range=(0,256))
        p_ = hist.astype(np.float32)/(hist.sum()+1e-6)
        ent = -np.sum(p_ * np.log2(p_ + 1e-12))
        lbp = float(np.clip((ent-5.0)/3.0, 0, 1))

        # GLCM suspicion
        levels=32
        q = (gray.astype(np.float32)/256.0*levels).astype(np.uint8)
        i = q[:,:-1].ravel(); j = q[:,1:].ravel()
        m = np.zeros((levels,levels), dtype=np.float64); np.add.at(m,(i,j),1)
        P = m/(m.sum()+1e-9)
        idx = np.arange(levels); di = (idx[:,None]-idx[None,:])**2
        contrast = float(np.sum(P*di)/((levels-1)**2))
        homog = float(np.sum(P/(1.0+di))); homog_n = np.clip(homog/max(homog,1.0), 0, 1)
        glcm = float(np.clip(0.6*contrast + 0.4*(1-homog_n), 0, 1))

        # Noise excess
        resid = gray.astype(np.float32)/255.0 - cv2.GaussianBlur(gray.astype(np.float32)/255.0,(5,5),0)
        noise_ex = float(np.clip((np.std(resid) - 0.05)/0.20, 0, 1))

        # Benford divergence on DCT
        h2,w2 = gray.shape; h2-=h2%8; w2-=w2%8
        if h2>0 and w2>0:
            blocks = gray[:h2,:w2].astype(np.float32).reshape(h2//8,8,w2//8,8).swapaxes(1,2).reshape(-1,8,8)
            vals = []
            for blk in blocks:
                dct = cv2.dct(blk); vals.append(np.abs(dct)[1:,1:].ravel())
            v = np.concatenate(vals); v=v[v>1e-6]
            if v.size>0:
                logs = np.log10(v); first = np.floor(10**(logs-np.floor(logs))).astype(int)
                hist = np.array([np.sum(first==d) for d in range(1,10)], dtype=np.float64)
                pp = hist/(hist.sum()+1e-9); ben = np.array([math.log10(1+1/d) for d in range(1,10)], dtype=np.float64)
                chi = np.sum((pp-ben)**2/(ben+1e-9))
                benf = float(np.clip(np.tanh(chi/2.5),0,1))
            else:
                benf = 0.0
        else:
            benf = 0.0

        varu = 0.0
        size=16; h3,w3 = gray.shape; h3-=h3%size; w3-=w3%size
        if h3>0 and w3>0:
            bl = gray[:h3,:w3].reshape(h3//size,size,w3//size,size).swapaxes(1,2).reshape(-1,size,size)
            vs = np.array([b.var() for b in bl], dtype=np.float32)+1e-6
            cv = float(np.std(vs)/np.mean(vs))
            varu = float(np.clip(1.0 - (cv-0.2)/0.6, 0, 1))

        feats = {"ELA":ela,"HF":hf,"Block":block,"ColorNoise":color,"LBP":lbp,"GLCM":glcm,"NoiseExcess":noise_ex,"Benford":benf,"VarUniform":varu}
        w = {"ELA":0.14,"HF":0.14,"Block":0.08,"ColorNoise":0.16,"LBP":0.12,"GLCM":0.12,"NoiseExcess":0.08,"Benford":0.10,"VarUniform":0.06}
        fused = sum(w[k]*feats[k] for k in w)
        if feats["NoiseExcess"]>0.55 and feats["ColorNoise"]>0.5: fused += 0.10
        if feats["Benford"]>0.6: fused += 0.07
        if feats["HF"]<0.25 and feats["Block"]<0.15 and feats["LBP"]<0.35: fused += 0.08
        return float(np.clip(fused,0,1)), feats

    # ---------- public API ----------
    def ensemble_status(self) -> str:
        return f"{len(self.ensemble)} learned model(s) loaded" if self.has_learned else "No learned weights found (artifact-only fallback)"

    def _face_crops(self, img_rgb: np.ndarray) -> List[Tuple[Image.Image, Tuple[int,int,int,int], float]]:
        boxes, probs, landmarks = self.face_detector.detect(img_rgb, landmarks=True)
        faces = []
        if boxes is None or len(boxes)==0:
            # fallback: use full frame
            pil = Image.fromarray(cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA))
            faces.append((pil, (0,0,img_rgb.shape[1], img_rgb.shape[0]), 1.0))
            return faces

        for b, p, lm in zip(boxes, probs, landmarks):
            if p is None or p < 0.85:  # quality gate
                continue
            x1,y1,x2,y2 = [int(v) for v in b]
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(img_rgb.shape[1],x2), min(img_rgb.shape[0],y2)
            if x2<=x1 or y2<=y1: continue
            crop = img_rgb[y1:y2, x1:x2]
            # align
            try:
                aligned = _affine_align(crop, lm - np.array([x1,y1]), out_size=256)
            except Exception:
                aligned = cv2.resize(crop, (256,256), interpolation=cv2.INTER_AREA)
            faces.append((Image.fromarray(aligned), (x1,y1,x2,y2), float(p)))
        if not faces:
            pil = Image.fromarray(cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA))
            faces.append((pil, (0,0,img_rgb.shape[1], img_rgb.shape[0]), 1.0))
        return faces

    def predict_face_prob(self, face_pil: Image.Image) -> Tuple[float, Dict[str, float]]:
        # artifact prob + learned ensemble blend
        art_p, feats = self._artifact_prob(face_pil)
        if self.has_learned:
            model_probs = []
            for name, m, size in self.ensemble:
                try:
                    model_probs.append(self._predict_model_prob(m, face_pil, size))
                except Exception:
                    continue
            if model_probs:
                p_model = float(np.mean(model_probs))
                feats["model_prob"] = p_model
                p = 0.8*p_model + 0.2*art_p
                return float(np.clip(p,0,1)), feats
        return art_p, feats

    def detect_image(self, image):
        pil = _to_pil(image)
        img = _ensure_rgb_np(pil)
        img = _resize_max(img, 900)

        faces = self._face_crops(img)
        contribs = []
        probs = []

        overlay = img.copy()
        for face_pil, (x1,y1,x2,y2), _ in faces:
            p, feats = self.predict_face_prob(face_pil)
            probs.append(p); contribs.append(feats)
            color = (0,0,255) if p>=0.5 else (0,200,0)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
            cv2.putText(overlay,f"{p*100:.1f}%",(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        prob = float(np.clip(np.max(probs) if probs else 0.0, 0, 1))
        is_fake = prob >= 0.5
        confidence = prob*100 if is_fake else (1.0-prob)*100

        heatmap = Image.fromarray(overlay)
        heatmap.info["contribs"] = contribs
        return is_fake, confidence, heatmap

    def score_frame_np(self, frame_rgb: np.ndarray) -> float:
        faces = self._face_crops(frame_rgb)
        ps = []
        for face_pil, _, _ in faces:
            p, _ = self.predict_face_prob(face_pil)
            ps.append(p)
        return float(np.clip(np.max(ps) if ps else 0.0, 0, 1))

    def detect_audio(self, audio_path):
        # placeholder; UI uses librosa viz separately
        return (False, 30.0, Image.new("RGB", (800, 300), (240, 240, 240)))
