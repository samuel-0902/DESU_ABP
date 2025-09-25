# Visualisation Graphviz de l’autoencodeur 1D :
# - Dessine l’encodeur/décodeur, le vecteur latent, la tête de classification,

import graphviz
import numpy as np
from IPython.display import display

# Helper pour les dimensions
def _ceil_div2(x): return (x + 1) // 2

def draw_ae_with_defaults(
    # --- Paramètres par défaut du script d'entraînement ---
    # Archi
    L_in=575,
    latent_dim=64,
    n_down=3,
    base_ch=16,
    dense_over=0.10,
    # Entraînement
    lr=3e-4,
    w_recon_init=1.0,
    w_cls_init=1.0,
    label_smoothing=0.0,
    steps_per_execution=512,
    epochs=30,
    patience=5,
    steps_cap=1000,
    # Données
    batch_size=512,
    frac_sup=0.25,
    # Paramètres fixes
    C_in=4,
    channels_recon=1,
    clipnorm=1.0,
    seed=42
):
    dot = graphviz.Digraph('AE_with_All_Defaults', format='svg')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.25', ranksep='0.35', fontname='Helvetica', fontsize='10')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9', arrowsize='0.7')

    # --- Calculs de dimensions (inchangés) ---
    L, C = L_in, base_ch
    enc_steps = [f"<tr><td>Conv→LayerNorm→ReLU</td><td>(B, {L}, {C})</td></tr>"]
    for i in range(n_down):
        L = _ceil_div2(L)
        C *= 2
        enc_steps.append(f"<tr><td>Down Block {i+1} (Conv→LN→ReLU ×2)</td><td>(B, {L}, {C})</td></tr>")
    L_down, C_down = L, C
    
    ch_bottleneck = C_down
    dec_start_ch = max(base_ch, ch_bottleneck // 2)
    dec_start_ch = int(np.ceil(dec_start_ch * (1.0 + float(dense_over))))
    L_dec, C_dec = L_down, dec_start_ch
    dec_rows = [f"<tr><td>Dense &rarr; Reshape</td><td>(B, {L_dec}, {C_dec})</td></tr>"]
    for i in range(n_down):
        L_dec *= 2
        C_dec = max(base_ch, C_dec // 2)
        dec_rows.append(f"<tr><td>Upsampling Step {i+1}</td><td>(B, {L_dec}, {C_dec})</td></tr>")
    L_logits, C_logits = L_dec, channels_recon

    # --- Graphe ---
    with dot.subgraph(name='cluster_archi') as c:
        c.attr(label="Architecture", color="transparent")
        c.node('input', f"Input\n(B, {L_in}, {C_in})", fillcolor='#eaf2f8')
        c.node('sanitize', 'FiniteSanitizer', fillcolor='#eaf2f8')
        encoder_table = f"""<
<table border="0" cellborder="1" cellspacing="0">
<tr><td bgcolor="#a9cce3" colspan="2"><b>Encodeur</b></td></tr>
{''.join(enc_steps)}
</table>>"""
        c.node('encoder_stack', encoder_table, shape='plaintext')
        c.node('gap', f"GlobalAveragePooling1D\n(B, {C_down})", fillcolor='#fdebd0')
        c.node('z', "Latent Vector (z)", shape='ellipse', fillcolor='#f9e79f', width='2')
        
        # ### LIGNE MODIFIÉE ###
        # Remplacé par un label générique pour être cohérent.
        c.node('cls_dense', "Tête de Classification\n(Bloc Dense)", fillcolor='#fadbd8')
        
        c.node('out_cls', "out_cls\n(B, 1)", shape='ellipse', fillcolor='#f5b7b1')
        c.node('loss_cls', "Loss Cls\n(BCE)", shape='note', fillcolor='#fde2e2')
        decoder_table = f"""<
<table border="0" cellborder="1" cellspacing="0">
<tr><td bgcolor="#d5f5e3" colspan="2"><b>Décodeur</b></td></tr>
{''.join(dec_rows)}
</table>>"""
        c.node('decoder_stack', decoder_table, shape='plaintext')
        c.node('conv_logits', f"Conv1D({channels_recon}, k=3)\n(B, {L_logits}, {C_logits})", fillcolor='#abebc6')
        c.node('crop', f"CropToRef\n(B, {L_in}, {C_logits})", fillcolor='#abebc6')
        c.node('out_recon', f"out_recon\n(B, {L_in}, {C_logits})", shape='ellipse', fillcolor='#82e0aa')
        c.node('loss_recon', "Loss Recon\n(masked MSE)", shape='note', fillcolor='#eafaf1')
        c.edge('input', 'sanitize')
        c.edge('sanitize', 'encoder_stack')
        c.edge('encoder_stack', 'gap')
        c.edge('gap', 'z')
        c.edge('z', 'cls_dense')
        c.edge('cls_dense', 'out_cls')
        c.edge('out_cls', 'loss_cls', style='dashed')
        c.edge('z', 'decoder_stack')
        c.edge('decoder_stack', 'conv_logits')
        c.edge('conv_logits', 'crop')
        c.edge('crop', 'out_recon')
        c.edge('out_recon', 'loss_recon', style='dashed')
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('cls_dense')
            s.node('decoder_stack')

    with dot.subgraph(name="cluster_optim") as c:
        c.attr(label="Optimisation", color="transparent", rank='sink')
        c.node('loss_total', "Loss Globale\nw_recon×L_recon + w_cls×L_cls", shape='note', fillcolor='#f5f5f5')
        c.node('optimizer', 'Optimiseur Adam', shape='cds', fillcolor='#f0f0f0')
        c.edge('loss_total', 'optimizer')
    dot.edge('loss_recon', 'loss_total', style='dashed', constraint='false')
    dot.edge('loss_cls', 'loss_total', style='dashed', constraint='false')

    with dot.subgraph(name="cluster_hparams") as hp:
        hp.attr(label="Hyperparamètres (défauts du script)", color="gray40", style="rounded", fontsize="9")
        
        arch_tbl = f"""<
<table BORDER="1" CELLBORDER="1" CELLSPACING="0">
<tr><td BGCOLOR="#e8daef" colspan="2"><b>Architecture</b></td></tr>
<tr><td>L_target</td><td>{L_in}</td></tr>
<tr><td>Latent Dim</td><td>{latent_dim}</td></tr>
<tr><td>Downsampling Steps</td><td>{n_down}</td></tr>
<tr><td>Base Channels</td><td>{base_ch}</td></tr>
<tr><td>Decoder Chan. Overprov.</td><td>{dense_over*100:.0f}%</td></tr>
</table>>"""
        
        train_tbl = f"""<
<table BORDER="1" CELLBORDER="1" CELLSPACING="0">
<tr><td BGCOLOR="#fff2cc" colspan="2"><b>Entraînement</b></td></tr>
<tr><td>Epochs / Patience</td><td>{epochs} / {patience}</td></tr>
<tr><td>Learning Rate</td><td>{lr}</td></tr>
<tr><td>Poids Pertes (init)</td><td>w_recon={w_recon_init}, w_cls={w_cls_init}</td></tr>
<tr><td>Label Smoothing</td><td>{label_smoothing}</td></tr>
<tr><td>Steps / Execution</td><td>{steps_per_execution}</td></tr>
<tr><td>Max Steps / Epoch</td><td>{steps_cap}</td></tr>
</table>>"""

        data_tbl = f"""<
<table BORDER="1" CELLBORDER="1" CELLSPACING="0">
<tr><td BGCOLOR="#ddeeff" colspan="2"><b>Données</b></td></tr>
<tr><td>Batch Size</td><td>{batch_size}</td></tr>
<tr><td>Fraction Supervisée</td><td>{frac_sup*100:.0f}%</td></tr>
<tr><td>Seed</td><td>{seed}</td></tr>
</table>>"""
        
        hp.node("T_ARCH", arch_tbl, shape="plaintext")
        hp.node("T_TRAIN", train_tbl, shape="plaintext")
        hp.node("T_DATA", data_tbl, shape="plaintext")
        hp.edge("T_ARCH", "T_TRAIN", style="invis")
        hp.edge("T_TRAIN", "T_DATA", style="invis")

    dot.edge('gap', 'T_ARCH', style='invis', constraint='false')
    
    return dot

# --- Génération et affichage ---
g = draw_ae_with_defaults()
display(g)
g.render('schema_architecture', format='png', view=True, cleanup=True)
