# -*- coding: utf-8 -*-
"""Nodes de cronometragem do Bruxos do VFX (fallback universal ao selo em tela).

O selo automatico (web/bruxos_node_timer.js) mede TODO node desenhando na tela.
Estes nodes sao o complemento pedido: um cronometro que voce PLUGA no fluxo pra
medir um trecho especifico -- funciona em qualquer modo de render (classico ou
Node 2.0), porque a medicao acontece no backend, nao no desenho.

Cronometro: passthrough que anota quando os dados passam por ele.
Relatorio: fecha e imprime a tabela de tempos (mais lento primeiro).
"""

import time


class _AnyType(str):
    def __ne__(self, other):
        return False


ANY = _AnyType("*")

_MARKS = []
_RUN_T0 = {"t": None}


def _fmt(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}m{s:04.1f}s"


class BruxosTimerMark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (ANY, {"tooltip": "Ligue aqui o dado que passa neste ponto (IMAGE, LATENT, MASK, MODEL, o que for). E repassado sem alteracao. Use isto pra medir nodes que nao mostram o selo automatico."}),
                "label": ("STRING", {"default": "etapa", "tooltip": "Nome deste ponto (ex.: 'depois do SAM3'). Aparece no relatorio e no console."}),
                "reset_run": ("BOOLEAN", {"default": False, "tooltip": "LIGADO no PRIMEIRO marco do fluxo: zera o cronometro e comeca a contar daqui."}),
            }
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("any", "elapsed")
    OUTPUT_TOOLTIPS = (
        "O mesmo dado que entrou, sem alteracao (passthrough).",
        "Tempo desde o marco anterior e desde o inicio da run, ja formatado.",
    )
    FUNCTION = "mark"
    CATEGORY = "Bruxos do VFX/Utilidades"
    DESCRIPTION = (
        "Cronometro passthrough. Intercale no fluxo pra medir um trecho especifico "
        "(complementa o selo automatico, util no Node 2.0). Segundos ate 60s, depois minutos. "
        "No 1o marco, ligue reset_run."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def mark(self, any, label, reset_run):
        global _MARKS
        now = time.time()
        if reset_run or _RUN_T0["t"] is None:
            _MARKS = []
            _RUN_T0["t"] = now
        prev = _MARKS[-1][1] if _MARKS else _RUN_T0["t"]
        _MARKS.append((label, now))
        delta = now - prev
        total = now - _RUN_T0["t"]
        print(f"[Bruxos Timer] {label}: +{_fmt(delta)}  (total {_fmt(total)})", flush=True)
        return (any, f"+{_fmt(delta)} | total {_fmt(total)}")


class BruxosTimerReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (ANY, {"tooltip": "Ligue a saida final do fluxo (ex.: o video que vai pro Save) pra garantir que o relatorio roda por ULTIMO."}),
            }
        }

    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("any", "report")
    OUTPUT_TOOLTIPS = ("Passthrough do dado final.", "Tabela: tempo de cada trecho + total + o mais lento.")
    FUNCTION = "report"
    CATEGORY = "Bruxos do VFX/Utilidades"
    DESCRIPTION = "Fecha a cronometragem e imprime a tabela: quanto cada trecho levou, o total e o mais lento. Ligue a saida final do workflow aqui."

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return time.time()

    def report(self, any):
        if not _MARKS or _RUN_T0["t"] is None:
            msg = "[Bruxos Timer] nenhum marco. Ponha 'Cronometro (Bruxos)' no fluxo."
            print(msg, flush=True)
            return (any, msg)
        t0 = _RUN_T0["t"]
        rows = []
        prev = t0
        slowest = ("", 0.0)
        for label, t in _MARKS:
            d = t - prev
            rows.append((label, d, t - t0))
            if d > slowest[1]:
                slowest = (label, d)
            prev = t
        total = _MARKS[-1][1] - t0
        width = max((len(r[0]) for r in rows), default=8)
        lines = ["", "=" * (width + 34), "  RELATORIO DE TEMPO - Bruxos do VFX", "=" * (width + 34),
                 f"  {'etapa'.ljust(width)}   trecho      acumulado", "  " + "-" * (width + 30)]
        for label, d, acc in rows:
            pct = (d / total * 100) if total > 0 else 0
            lines.append(f"  {label.ljust(width)}   {_fmt(d).rjust(8)}   {_fmt(acc).rjust(9)}   {pct:4.0f}%")
        lines.append("  " + "-" * (width + 30))
        lines.append(f"  {'TOTAL'.ljust(width)}   {_fmt(total).rjust(8)}")
        lines.append(f"  mais lento: {slowest[0]} ({_fmt(slowest[1])})")
        lines.append("=" * (width + 34))
        report = "\n".join(lines)
        print(report, flush=True)
        return (any, report)


NODE_CLASS_MAPPINGS = {
    "BruxosTimerMark": BruxosTimerMark,
    "BruxosTimerReport": BruxosTimerReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BruxosTimerMark": "Cronometro (Bruxos)",
    "BruxosTimerReport": "Relatorio de Tempo (Bruxos)",
}
