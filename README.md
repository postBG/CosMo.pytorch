# CoSMo.pytorch

Install torch and torchvision via following command (CUDA10)

```bash
pip install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/
whl/torch_stable.html
```

Install other packages
```bash
pip install -r requirements.txt
```

Download NLTK punkt
```python
import nltk
nltk.download('punkt')
```