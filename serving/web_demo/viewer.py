"""An isolated crystal viewer fed only validated numeric structure data."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymatgen.core import Structure


THREEDMOL_URL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"


def structure_payload(structure: "Structure") -> dict:
    """Serialize atoms and lattice vectors; never embed uploaded CIF text."""
    atoms = []
    for index, site in enumerate(structure):
        element = site.specie.symbol
        if not re.fullmatch(r"[A-Z][a-z]?", element):
            raise ValueError("The viewer requires ordinary chemical elements.")
        fractional = [float(value) % 1.0 for value in site.frac_coords]
        xyz = [float(value) for value in structure.lattice.get_cartesian_coords(fractional)]
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("Cannot display non-finite atomic coordinates.")
        atoms.append({"elem": element, "x": xyz[0], "y": xyz[1], "z": xyz[2], "serial": index + 1})
    lattice = [[float(value) for value in row] for row in structure.lattice.matrix]
    if not all(math.isfinite(value) for row in lattice for value in row):
        raise ValueError("Cannot display a non-finite lattice.")
    return {"atoms": atoms, "lattice": lattice}


def crystal_html(structure: "Structure", height: int = 360) -> str:
    """Return an iframe document with rotate, zoom, reset and spin controls."""
    payload = json.dumps(structure_payload(structure), allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _TEMPLATE.replace("__PAYLOAD__", payload).replace("__CDN__", THREEDMOL_URL).replace("__HEIGHT__", str(int(height)))


def render_crystal(structure: "Structure", height: int = 360) -> None:
    import streamlit as st

    st.iframe(crystal_html(structure, height), height=height + 80, alt="Interactive crystal structure")


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;color:#20233D;font:12px system-ui,sans-serif}
.shell{border:1px solid #E7E7EF;border-radius:12px;overflow:hidden;background:#FCFCFE}
#viewport{position:relative;width:100%;height:__HEIGHT__px}
.toolbar{display:flex;align-items:center;gap:8px;padding:10px 12px;border-top:1px solid #EEEEF4;background:white;flex-wrap:wrap}
button{background:#FFF;border:1px solid #DADAE6;border-radius:6px;padding:5px 9px;color:#2C2F6E;font:inherit;cursor:pointer}
button:hover{background:#F3F3FA}button:focus-visible{outline:2px solid #B76B43;outline-offset:2px}
.hint{color:#707385;margin-right:auto}.legend{display:flex;gap:10px;flex-wrap:wrap}
.element{display:inline-flex;align-items:center;gap:4px}.dot{width:9px;height:9px;border-radius:50%;border:1px solid #0002}
#fallback{position:absolute;inset:0;display:grid;place-items:center;padding:30px;text-align:center;color:#707385}
@media(max-width:380px){.hint{width:100%}}
</style></head><body><div class="shell">
<div id="viewport" role="img" aria-label="Interactive crystal structure. Drag to rotate, scroll to zoom."><div id="fallback">Loading crystal viewer…</div></div>
<div class="toolbar"><span class="hint">Drag to rotate · Scroll to zoom</span><button id="reset" type="button">Reset</button><button id="spin" type="button" aria-pressed="false">Spin</button><div id="legend" class="legend"></div></div>
</div>
<script>
const data = __PAYLOAD__;
const failure = () => {const fallback=document.getElementById('fallback');if(fallback)fallback.textContent = 'The 3D viewer could not load. You can still download the CIF.';};
const script = document.createElement('script'); script.src = '__CDN__'; script.onerror = failure;
script.onload = () => {try {
  const viewer = $3Dmol.createViewer(document.getElementById('viewport'), {backgroundColor:'#FCFCFE',antialias:true});
  const model = viewer.addModel(); model.addAtoms(data.atoms);
  viewer.setStyle({}, {sphere:{scale:0.28,colorscheme:'Jmol'}});
  const vector = arr => ({x:arr[0],y:arr[1],z:arr[2]});
  const vertices = Array.from({length:8}, (_, bits) => {
    const v = [0,0,0];
    for(let axis=0;axis<3;axis++) if(bits & (1<<axis)) for(let j=0;j<3;j++) v[j] += data.lattice[axis][j];
    return vector(v);
  });
  const axes = ['a','b','c']; const axisColors = ['#2C2F6E','#B76B43','#607585'];
  for(let bits=0;bits<8;bits++) for(let axis=0;axis<3;axis++) if(!(bits & (1<<axis))) {
    viewer.addLine({start:vertices[bits],end:vertices[bits | (1<<axis)],color:bits===0?axisColors[axis]:'#ADB0C3',linewidth:1.4});
  }
  axes.forEach((axis,i) => viewer.addLabel(axis,{position:vertices[1<<i],fontColor:axisColors[i],backgroundOpacity:0,fontSize:13,showBackground:false,inFront:true}));
  let hovered = null;
  viewer.setHoverable({}, true, (atom) => {
    hovered = viewer.addLabel(atom.elem + ' · site ' + atom.serial,{position:atom,fontSize:12,fontColor:'#20233D',backgroundColor:'white',backgroundOpacity:.95,borderThickness:1,borderColor:'#DADAE6'});
    viewer.render();
  }, () => {if(hovered){viewer.removeLabel(hovered);hovered=null;} viewer.render();});
  const bounds = viewer.addModel(); bounds.addAtoms(vertices.map(p => ({...p,elem:'X'}))); bounds.setStyle({},{});
  viewer.zoomTo(); viewer.rotate(18,'x'); viewer.rotate(-20,'y'); viewer.zoom(1.18); viewer.render();
  const initial = viewer.getView(); let spinning = false;
  document.getElementById('reset').onclick = () => {viewer.setView(initial);viewer.render();};
  document.getElementById('spin').onclick = event => {spinning=!spinning;viewer.spin(spinning?'y':false,.5,true);event.target.textContent=spinning?'Pause':'Spin';event.target.setAttribute('aria-pressed',String(spinning));};
  const palette = $3Dmol.elementColors.Jmol || $3Dmol.elementColors.defaultColors;
  [...new Set(data.atoms.map(atom => atom.elem))].sort().forEach(elem => {
    const item=document.createElement('span');item.className='element';
    const dot=document.createElement('span');dot.className='dot';dot.style.background='#'+(palette[elem] ?? 0x909090).toString(16).padStart(6,'0');
    item.appendChild(dot);item.appendChild(document.createTextNode(elem));document.getElementById('legend').appendChild(item);
  });
  new ResizeObserver(() => {viewer.resize();viewer.render();}).observe(document.getElementById('viewport'));
  document.getElementById('fallback').remove();
} catch(error) {failure();}};
document.head.appendChild(script);
</script></body></html>
"""
