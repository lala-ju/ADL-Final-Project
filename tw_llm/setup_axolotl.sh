git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'

pip install flash-attn==2.3.6
