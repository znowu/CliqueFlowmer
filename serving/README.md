---
title: CliqueFlowmer
emoji: 🧪
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

For the maintained CPU-only upload → optimize → visualize demo, see
[web_demo/README.md](web_demo/README.md). It uses its own dependency lock and can
be deployed on free Streamlit Community Cloud hosting.

The Gradio app in this directory is the legacy demo for the paper
**Offline Materials Optimization with CliqueFlowmer**.

This Space supports:
- CIF → latent
- CIF → latent → reconstruction
- latent → CIF

Paper: https://huggingface.co/papers/2603.06082
Code: https://github.com/znowu/CliqueFlowmer
