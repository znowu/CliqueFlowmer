"""Public, single-crystal CPU demo. Run with ``streamlit run``."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import streamlit as st

# Streamlit executes an entrypoint as a script, including when deployed from a
# subdirectory. Keep repository imports independent of the launching directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.web_demo.service import DemoService, EXAMPLES, MAX_CIF_BYTES, RESULT_TTL_SECONDS, validate_cif
from serving.web_demo.viewer import render_crystal


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
BUDGETS = {"Quick · 100 steps": "quick", "Standard · 500 steps": "standard"}


@st.cache_resource(show_spinner=False)
def get_service() -> DemoService:
    """Share one bounded worker and one model across browser sessions."""
    return DemoService()


def _init_state() -> None:
    for key, value in {
        "cf_input_digest": None,
        "cf_preview": None,
        "cf_input_error": None,
        "cf_result": None,
        "cf_result_created": None,
        "cf_output_preview": None,
        "cf_job": None,
        "cf_cancel_requested": False,
        "cf_error": None,
        "cf_notice": None,
        "cf_operation": None,
    }.items():
        st.session_state.setdefault(key, value)


def _prepare_input(cif_text: str, input_error: str | None = None) -> None:
    digest = hashlib.sha256((cif_text + "\0" + (input_error or "")).encode("utf-8")).hexdigest()
    if digest == st.session_state.cf_input_digest:
        return
    job = st.session_state.cf_job
    if job is not None:
        job.cancel()
    st.session_state.update(
        cf_input_digest=digest,
        cf_preview=None,
        cf_output_preview=None,
        cf_result=None,
        cf_result_created=None,
        cf_job=None,
        cf_cancel_requested=False,
        cf_error=None,
        cf_notice="Input changed; the previous run was cancelled." if job is not None else None,
        cf_input_error=input_error,
        cf_operation=None,
    )
    if input_error or not cif_text.strip():
        return
    if len(cif_text.encode("utf-8")) > MAX_CIF_BYTES:
        st.session_state.cf_input_error = "Please use a CIF file smaller than 100 KB."
        return
    try:
        st.session_state.cf_preview = validate_cif(cif_text)
    except ValueError as exc:
        st.session_state.cf_input_error = str(exc)


def _start(cif_text: str, operation: str, seed: int, budget: str) -> None:
    try:
        job = get_service().submit(cif_text, operation=operation, seed=seed, budget=budget)
    except ValueError as exc:
        st.session_state.cf_error = str(exc)
        return
    st.session_state.update(
        cf_job=job,
        cf_result=None,
        cf_result_created=None,
        cf_output_preview=None,
        cf_error=None,
        cf_notice=None,
        cf_cancel_requested=False,
        cf_operation=operation,
    )
    st.rerun()


@st.fragment(run_every=1)
def _job_panel() -> None:
    """Poll only the small status panel, preserving the interactive viewers."""
    job = st.session_state.cf_job
    if job is None:
        return
    snapshot = job.snapshot()
    status = snapshot.get("status", "queued")
    if status in TERMINAL_STATUSES:
        st.session_state.cf_job = None
        st.session_state.cf_cancel_requested = False
        if status == "completed":
            result = snapshot.get("result")
            if result is None:
                st.session_state.cf_error = "The run finished without a result. Please try again."
            else:
                st.session_state.cf_result = result
                st.session_state.cf_result_created = time.monotonic()
                output_cif = result.get("output_cif")
                if output_cif:
                    try:
                        st.session_state.cf_output_preview = validate_cif(output_cif)
                    except ValueError:
                        # Downloads and latent diagnostics remain available even
                        # if a generated cell cannot be rendered as a valid input.
                        st.session_state.cf_output_preview = None
        elif status == "failed":
            st.session_state.cf_error = str(snapshot.get("error") or "The run could not finish. Please try another structure.")
        else:
            st.session_state.cf_notice = "Run cancelled. You can start again with the same structure."
        st.rerun()
    progress = min(1.0, max(0.0, float(snapshot.get("progress", 0))))
    message = snapshot.get("message") or "Preparing your crystal…"
    st.progress(progress, text=message)
    if st.button("Cancel run", key="cf_cancel", disabled=st.session_state.cf_cancel_requested):
        job.cancel()
        st.session_state.cf_cancel_requested = True
    if st.session_state.cf_cancel_requested:
        st.caption("Cancelling at the next computation checkpoint…")
    else:
        st.caption("Runs on a shared CPU. Keep this page open while the calculation finishes.")


def _structure_summary(structure, summary: dict | None = None) -> None:
    summary = summary or {}
    formula = summary.get("formula", structure.composition.reduced_formula)
    sites = int(summary.get("n_sites", len(structure)))
    volume = float(summary.get("volume", structure.volume))
    st.markdown(f"**{formula}**")
    st.caption(f"{sites} atom{'s' if sites != 1 else ''} in the cell · {volume:.2f} Å³")


def _placeholder(title: str, message: str) -> None:
    # Only fixed application text is passed here.
    st.markdown(
        '<div class="cf-empty"><div class="cf-orbit">◇</div>'
        f'<strong>{title}</strong><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def _structures() -> None:
    original = st.session_state.cf_preview
    result = st.session_state.cf_result
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("#### Input crystal")
        if original is not None:
            _structure_summary(original, (result or {}).get("input_summary"))
            render_crystal(original)
        else:
            _placeholder("Start with a crystal", "Upload a CIF, paste its contents, or select an example.")
    with right:
        operation = st.session_state.cf_operation
        label = {"reconstruct": "Reconstruction", "encode": "Latent representation"}.get(operation, "Optimized candidate")
        st.markdown(f"#### {label}")
        output = st.session_state.cf_output_preview
        if result is not None and output is not None:
            _structure_summary(output, result.get("output_summary"))
            render_crystal(output)
            st.caption("Unrelaxed model output. Optimization can change both composition and geometry.")
        elif result is not None and result.get("output_cif"):
            st.info("A candidate was generated, but it could not be displayed. Its CIF is available below.")
        elif result is not None:
            _placeholder("Representation ready", f"Encoded into {len(result.get('latent', []))} latent coordinates. Download the representation below.")
        elif st.session_state.cf_job is not None:
            _placeholder("Exploring latent space", "Your result will appear here when the run finishes.")
        else:
            _placeholder("A new candidate starts here", "Optimize the input, or reconstruct it from its representation.")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _npy_bytes(values) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    return buffer.getvalue()


@st.fragment(run_every=60)
def _expire_result() -> None:
    created = st.session_state.cf_result_created
    if created is not None and time.monotonic() - created >= RESULT_TTL_SECONDS:
        st.session_state.update(cf_result=None, cf_result_created=None, cf_output_preview=None)
        st.session_state.cf_notice = "The previous result expired after 30 minutes. Run the demo again to recreate it."
        st.rerun()


def _clique_heatmap(values) -> None:
    """The released checkpoint uses the unpermuted 8 × 16 clique chain."""
    if len(values) != 121:
        return
    import altair as alt

    rows = [
        {"Clique": clique + 1, "Position": position + 1,
         "Latent coordinate": clique * 15 + position,
         "Value": float(values[clique * 15 + position])}
        for clique in range(8) for position in range(16)
    ]
    extent = max(1e-6, max(abs(row["Value"]) for row in rows))
    chart = alt.Chart(alt.Data(values=rows)).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("Position:O", title="Coordinate within clique", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Clique:O", title="Clique"),
        color=alt.Color("Value:Q", scale=alt.Scale(domain=[-extent, 0, extent], range=["#2C2F6E", "#F5F5FA", "#B76B43"])),
        tooltip=[alt.Tooltip("Clique:O"), alt.Tooltip("Position:O"),
                 alt.Tooltip("Latent coordinate:Q"), alt.Tooltip("Value:Q", format=".9g")],
    ).properties(height=210).configure_view(stroke=None)
    st.altair_chart(chart, width="stretch")
    st.caption("8 cliques × 16 coordinates. Adjacent cliques share one endpoint coordinate, giving 121 distinct coordinates. The colors show latent values, not chemical properties.")


def _result_details(result: dict) -> None:
    metadata = result.get("metadata") or {}
    property_name = metadata.get("property_name", "Latent objective")
    unit = metadata.get("unit", "model score")
    initial = result.get("initial_score")
    final = result.get("final_score")
    if initial is not None:
        st.markdown("#### Model predictions")
        columns = st.columns(3)
        columns[0].metric("Starting score", f"{float(initial):.6f}")
        if final is not None:
            columns[1].metric("Final score", f"{float(final):.6f}")
            columns[2].metric("Change", f"{float(final) - float(initial):+.6f}")
        st.caption(f"{property_name} · {unit} · lower is the optimization target")
        note = metadata.get("objective_note") or metadata.get("note")
        st.caption(note or "These are predictions of the latent model, not DFT energies or a confirmation of physical stability.")

    history = result.get("history") or []
    if len(history) > 1:
        import altair as alt

        chart = alt.Chart(alt.Data(values=history)).mark_line(color="#B76B43", strokeWidth=2).encode(
            x=alt.X("step:Q", title="Optimization step", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("score:Q", title="Model score" if unit == "model score" else f"Score ({unit})", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("step:Q", title="Step"), alt.Tooltip("score:Q", title="Score", format=".9g")],
        ).properties(height=230).configure_view(stroke=None).configure_axis(gridColor="#EEEEF4")
        st.altair_chart(chart, width="stretch")

    st.markdown("#### Take the result with you")
    st.caption("CIF structures and full-precision representations, with settings and model provenance.")
    columns = st.columns(4)
    if result.get("output_cif"):
        columns[0].download_button(
            "Download candidate CIF" if st.session_state.cf_operation == "optimize" else "Download reconstruction CIF",
            result["output_cif"], file_name="cliqueflowmer_candidate.cif" if st.session_state.cf_operation == "optimize" else "cliqueflowmer_reconstruction.cif", mime="chemical/x-cif",
            width="stretch",
        )
    else:
        columns[0].download_button("Download input CIF", result["input_cif"], file_name="input.cif", mime="chemical/x-cif", width="stretch")
    columns[1].download_button(
        "Encoded latent · NPY", _npy_bytes(result["latent"]),
        file_name="cliqueflowmer_latent.npy", mime="application/octet-stream", width="stretch",
    )
    if result.get("optimized_latent") is not None:
        columns[2].download_button(
            "Optimized latent · NPY", _npy_bytes(result["optimized_latent"]),
            file_name="cliqueflowmer_optimized_latent.npy", mime="application/octet-stream", width="stretch",
        )
    columns[3].download_button(
        "Complete result · JSON", json.dumps(_plain(result), indent=2, allow_nan=False),
        file_name="cliqueflowmer_result.json", mime="application/json", width="stretch",
    )
    with st.expander("Representation and run details"):
        st.caption(f"{len(result.get('latent', []))} latent coordinates. Download files preserve every reported digit.")
        if result.get("optimized_latent") is not None:
            selection = st.radio("Representation to view", ["Encoded", "Optimized"], horizontal=True, key="cf_latent_view")
            _clique_heatmap(result["optimized_latent"] if selection == "Optimized" else result["latent"])
        else:
            _clique_heatmap(result["latent"])
        latent_columns = {"Coordinate": list(range(len(result["latent"]))), "Encoded": result["latent"]}
        if result.get("optimized_latent") is not None:
            latent_columns["Optimized"] = result["optimized_latent"]
        st.dataframe(latent_columns, hide_index=True, width="stretch")
        st.json(_plain(metadata), expanded=False)


def main() -> None:
    st.set_page_config(page_title="CliqueFlowmer · Crystal explorer", page_icon=":material/hub:", layout="wide")
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="cf-header"><span class="cf-kicker">CRYSTAL REPRESENTATION &amp; DESIGN</span>'
        '<h1>CliqueFlowmer<span class="cf-period">.</span></h1>'
        '<p>Explore a crystal’s representation. Reconstruct it. Find a new candidate.</p></div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        input_column, action_column = st.columns([1.8, 1], gap="medium")
        with input_column:
            modes = ["Example", "Upload CIF", "Paste CIF"] if EXAMPLES else ["Upload CIF", "Paste CIF"]
            mode = st.radio("Choose a structure", modes, horizontal=True, key="cf_input_mode")
            cif_text, input_error = "", None
            if mode == "Example":
                name = st.selectbox("Example crystal", list(EXAMPLES), key="cf_example")
                cif_text = EXAMPLES[name]
            elif mode == "Upload CIF":
                uploaded = st.file_uploader("CIF file · up to 100 KB", type=["cif"], accept_multiple_files=False, key="cf_upload")
                if uploaded is not None:
                    raw = uploaded.getvalue()
                    if len(raw) > MAX_CIF_BYTES:
                        input_error = "Please use a CIF file smaller than 100 KB."
                    else:
                        try:
                            cif_text = raw.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            input_error = "The CIF must be a UTF-8 text file."
            else:
                cif_text = st.text_area("CIF contents", height=160, placeholder="data_crystal\n_cell_length_a …", key="cf_paste")
            _prepare_input(cif_text, input_error)
            st.caption("One ordered crystal · 1–20 atoms in the supplied cell")
            if st.session_state.cf_input_error:
                st.warning(st.session_state.cf_input_error)

        with action_column:
            st.markdown("**Explore this crystal**")
            st.caption("Target: minimize formation energy per atom (eV/atom). Displayed scores use the model’s internal scale.")
            with st.expander("Run settings"):
                budget_label = st.selectbox("Optimization budget", list(BUDGETS), key="cf_budget")
                seed = int(st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1, key="cf_seed"))
                st.caption("Quick uses 100 optimization steps; Standard uses 500. The seed makes a run repeatable in the same environment.")
            disabled = st.session_state.cf_preview is None or st.session_state.cf_job is not None
            if st.button("Optimize crystal", type="primary", width="stretch", disabled=disabled):
                _start(cif_text, "optimize", seed, BUDGETS[budget_label])
            secondary = st.columns(2)
            if secondary[0].button("Encode", width="stretch", disabled=disabled, help="Get the latent representation without generating a structure."):
                _start(cif_text, "encode", seed, BUDGETS[budget_label])
            if secondary[1].button("Reconstruct", width="stretch", disabled=disabled, help="Decode a structure from the input's latent representation."):
                _start(cif_text, "reconstruct", seed, BUDGETS[budget_label])
            st.caption("Generated candidates are unrelaxed. This demo does not run DFT.")

    if st.session_state.cf_error:
        st.error(st.session_state.cf_error)
    if st.session_state.cf_notice:
        st.info(st.session_state.cf_notice)
    if st.session_state.cf_job is not None:
        _job_panel()
    _structures()
    if st.session_state.cf_result is not None:
        _expire_result()
        _result_details(st.session_state.cf_result)
    st.divider()
    st.caption(
        "[Paper](https://arxiv.org/abs/2603.06082) · "
        "[Code](https://github.com/znowu/CliqueFlowmer) · "
        "[Model weights](https://huggingface.co/iamkuba/CliqueFlowmer) · "
        "Crystal visualization with [3Dmol.js](https://3dmol.org/)"
    )


_CSS = """<style>
.block-container{max-width:1200px;padding-top:2.3rem;padding-bottom:2rem}
.cf-header{padding:0 0 1.4rem}.cf-kicker{font-size:.7rem;font-weight:650;letter-spacing:.15em;color:#B76B43}
.cf-header h1{font-size:3.2rem;letter-spacing:-.055em;font-weight:700;color:#2C2F6E;margin:.12rem 0 .3rem;padding:0;line-height:1.13}
.cf-period{color:#B76B43}.cf-header p{font-size:1.05rem;color:#727487;margin:0;line-height:1.6}
h4{color:#2C2F6E;letter-spacing:-.02em;padding-top:1.1rem!important}
[data-testid="stMetricValue"]{font-size:1.7rem;color:#2C2F6E}
[data-testid="stButton"] button{border-radius:8px;min-height:2.65rem}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-radius:12px}
.cf-empty{height:406px;border:1px dashed #DADBE8;border-radius:12px;background:linear-gradient(145deg,#FCFCFE,#F7F7FB);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:30px;color:#2C2F6E;margin-top:80px}
.cf-empty p{color:#85879A;font-size:.9rem;max-width:260px;line-height:1.6}.cf-orbit{color:#B76B43;font-size:70px;font-weight:200;line-height:1.2;margin-bottom:14px}
@media(max-width:640px){.block-container{padding-top:2.5rem}.cf-header h1{font-size:2.5rem}.cf-empty{height:240px;margin-top:0}}
</style>"""


if __name__ == "__main__":
    main()
