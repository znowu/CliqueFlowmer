from __future__ import annotations

import os
import traceback

import gradio as gr
from pymatgen.core import Structure

from inference import (
    decode_latent,
    encode_structure,
    latent_to_text,
    load_model_from_hf,
    reconstruct_structure,
    structure_from_cif_text,
    structure_to_cif_text,
    text_to_latent,
)

HF_REPO_ID = os.environ.get("CLIQUEFLOWMER_HF_REPO_ID", "znowu/CliqueFlowmer-weights")
HF_FILENAME = os.environ.get("CLIQUEFLOWMER_HF_FILENAME", "cliqueflowmer_mp20.pth")
MODEL_DEVICE = os.environ.get("CLIQUEFLOWMER_DEVICE", "cpu")

model = load_model_from_hf(
    repo_id=HF_REPO_ID,
    filename=HF_FILENAME,
    device=MODEL_DEVICE,
    strict=True,
)


def _formula(structure: Structure) -> str:
    try:
        return structure.composition.reduced_formula
    except Exception:
        return "Unknown"


def encode_cif(cif_text: str):
    try:
        structure = structure_from_cif_text(cif_text)
        z = encode_structure(model, structure)
        return _formula(structure), latent_to_text(z), ""
    except Exception:
        return None, None, traceback.format_exc()


def reconstruct_cif(cif_text: str, integration: str):
    try:
        structure = structure_from_cif_text(cif_text)
        z, recon = reconstruct_structure(model, structure, integration=integration)
        return (
            _formula(structure),
            _formula(recon),
            latent_to_text(z),
            structure_to_cif_text(recon),
            "",
        )
    except Exception:
        return None, None, None, None, traceback.format_exc()


def decode_from_latent(latent_text: str, integration: str):
    try:
        z = text_to_latent(latent_text)
        structure = decode_latent(model, z, integration=integration)
        return _formula(structure), structure_to_cif_text(structure), ""
    except Exception:
        return None, None, traceback.format_exc()


with gr.Blocks(title="CliqueFlowmer") as demo:
    gr.Markdown(
        """
# CliqueFlowmer
Lightweight public demo.

Supports:
- CIF → latent
- CIF → latent → reconstruction
- latent → CIF
"""
    )

    with gr.Tab("Encode CIF"):
        cif_in = gr.Textbox(label="Paste CIF", lines=18)
        btn = gr.Button("Encode")
        formula_out = gr.Textbox(label="Formula")
        latent_out = gr.Textbox(label="Latent vector", lines=8)
        err_out = gr.Textbox(label="Error", lines=8)
        btn.click(encode_cif, inputs=[cif_in], outputs=[formula_out, latent_out, err_out])

    with gr.Tab("Reconstruct CIF"):
        cif_in2 = gr.Textbox(label="Paste CIF", lines=18)
        integration2 = gr.Dropdown(
            choices=["cfg", "euler", "dopri5", "rk4"],
            value="cfg",
            label="Geometry integration",
        )
        btn2 = gr.Button("Reconstruct")
        input_formula = gr.Textbox(label="Input formula")
        output_formula = gr.Textbox(label="Reconstructed formula")
        latent2 = gr.Textbox(label="Latent vector", lines=8)
        cif_out2 = gr.Textbox(label="Reconstructed CIF", lines=18)
        err2 = gr.Textbox(label="Error", lines=8)
        btn2.click(
            reconstruct_cif,
            inputs=[cif_in2, integration2],
            outputs=[input_formula, output_formula, latent2, cif_out2, err2],
        )

    with gr.Tab("Decode latent"):
        latent_in3 = gr.Textbox(label="Comma-separated latent vector", lines=8)
        integration3 = gr.Dropdown(
            choices=["cfg", "euler", "dopri5", "rk4"],
            value="cfg",
            label="Geometry integration",
        )
        btn3 = gr.Button("Decode")
        formula3 = gr.Textbox(label="Decoded formula")
        cif_out3 = gr.Textbox(label="Decoded CIF", lines=18)
        err3 = gr.Textbox(label="Error", lines=8)
        btn3.click(
            decode_from_latent,
            inputs=[latent_in3, integration3],
            outputs=[formula3, cif_out3, err3],
        )

demo.launch()