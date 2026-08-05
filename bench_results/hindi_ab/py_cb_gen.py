import os, sys
import mlx.core as mx, numpy as np, soundfile as sf
from mlx_audio.tts.utils import load_model, get_model_path
from mlx_audio.tts.models.chatterbox.tokenizer import MTLTokenizer
m = load_model(get_model_path("/tmp/chatterbox-4bit"))
tok = MTLTokenizer("/tmp/chatterbox-4bit/tokenizer.json")
text = os.environ["PY_TEXT"]
tt = tok.text_to_tokens(text, language_id="hi")
toks = m.t3.inference(t3_cond=m._conds.t3, text_tokens=tt, max_new_tokens=150, cfg_weight=0.5)
mx.eval(toks)
wav, _ = m.s3gen.inference(toks, m._conds.gen)
mx.eval(wav)
sf.write(os.environ["PY_OUT"], np.array(wav[0]), 24000)
print("  py tokens:", toks.shape[1], "wav:", len(np.array(wav[0])), file=sys.stderr)
