# -*- coding: utf-8 -*-
r"""
Bruxos do VFX — Block Swap (RAM Offload) pro Wan 2.2 / Bernini-R nativo
======================================================================
Mantem os primeiros N blocos do transformer (DiT) residentes na RAM (opcional
pinned) e faz streaming de cada bloco pra GPU so durante o forward dele. Os
patches de LoRA sao BAKED nos pesos na CPU uma vez no load, entao o caminho
lento de cast por-camada (LowVramPatch) e evitado. Roda modelos maiores que a
VRAM (Wan/Bernini-R 14B fp16 ~28.6GB numa placa de 24GB) com custo ~0 quando o
gargalo e compute (alta resolucao).

PORTADO fielmente do ComfyUI-JITBlockSwap (lovemachine100) — mesma logica
verificada no Wan/Bernini-R, so com nomes/tooltips Bruxos. Credito ao autor.

REQUISITO: lance o ComfyUI com `--disable-dynamic-vram` (ModelPatcher legado).
Com DynamicVRAM ligado o node e NO-OP (o proprio DynamicVRAM cuida da colocacao)
e avisa no console `[Bruxos BlockSwap] dynamic VRAM patcher detected, skipping`.

Cadeia: Loader -> (LoRA) -> BlockSwap -> I2V/Sampler.
EXPERIMENTAL: codigo de memoria de baixo nivel; teste com cuidado na sua GPU.
"""

import functools
import logging

try:
    import torch
    import comfy.model_management as mm
    import comfy.model_patcher
    from comfy.patcher_extension import CallbacksMP
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

log = logging.getLogger(__name__)
CAT = "Bruxos do VFX/Bernini"
CALLBACK_KEY = "bruxos_blockswap_ram_offload"

# folga transiente de VRAM pro(s) bloco(s) em voo durante o swap
_SWAP_BUFFER_EXTRA = 128 * 1024 * 1024


def _make_swap_forward(block):
    orig_forward = block._bs_orig_forward

    def swap_forward(*args, **kwargs):
        state = getattr(block, "_bs_state", None)
        if state is None or not state.get("active", False):
            return orig_forward(*args, **kwargs)
        load_device = state["load_device"]
        masters = state["masters"]
        non_blocking = not state.get("sync_transfers", True)
        for t, master in masters:
            t.data = master.to(load_device, non_blocking=non_blocking)
        if not non_blocking:
            # serializa a fila antes do compute: H2D pinned async intercalado
            # com kernels DiT trava a GPU no Windows/WDDM (100% util em idle).
            torch.cuda.synchronize()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            for t, master in masters:
                t.data = master

    return swap_forward


def _iter_block_tensors(block):
    for p in block.parameters(recurse=True):
        yield p
    for b in block.buffers(recurse=True):
        yield b


def _finalize_module(patcher, name, module, target_device, unpin_all=False):
    """Bakeia os patches pendentes (LoRA) dos params folha no target_device e
    remove o caminho de cast lowvram. Idempotente.

    unpin_all=True sempre que o modulo vai sair da CPU: o comfy pina os pesos
    offloadados (cudaHostRegister); um tensor pinned liberado ainda registrado
    deixa um registro pendente que vira CUDA error: invalid argument depois."""
    params = dict(module.named_parameters(recurse=False))
    for pname in params:
        key = "{}.{}".format(name, pname) if name else pname
        if unpin_all or key in patcher.patches:
            patcher.unpin_weight(key)
    if getattr(module, "comfy_patched_weights", False) is not True:
        for pname in params:
            key = "{}.{}".format(name, pname) if name else pname
            if key in patcher.patches:
                patcher.patch_weight_to_device(key, device_to=target_device)
        if params or hasattr(module, "comfy_cast_weights"):
            module.comfy_patched_weights = True
    comfy.model_patcher.wipe_lowvram_weight(module)


def _finalize_tree(patcher, prefix, root, target_device, unpin_all=False):
    for sub_name, m in root.named_modules():
        full = "{}.{}".format(prefix, sub_name) if sub_name else prefix
        _finalize_module(patcher, full, m, target_device, unpin_all=unpin_all)
    root.to(target_device)


def _deactivate_block(block, unpin=True):
    state = getattr(block, "_bs_state", None)
    if state is None:
        return
    state["active"] = False
    if unpin:
        for t in state.get("pinned", []):
            mm.unpin_memory(t)
        state["pinned"] = []
    block._bs_state = None


def _on_load(patcher, device_to, lowvram_model_memory, force_patch_weights, full_load,
             blocks_to_swap=0, pin_masters=True):
    try:
        if patcher.is_dynamic():
            log.warning("[Bruxos BlockSwap] dynamic VRAM patcher detected, skipping. "
                        "Lance o ComfyUI com --disable-dynamic-vram pra usar o block swap.")
            return
        base = patcher.model
        dm = getattr(base, "diffusion_model", None)
        blocks = getattr(dm, "blocks", None)
        if blocks is None:
            log.warning("[Bruxos BlockSwap] modelo sem diffusion_model.blocks, pulando.")
            return

        offload_device = patcher.offload_device
        load_device = torch.device(device_to if device_to is not None else patcher.load_device)
        if not mm.is_device_cuda(load_device):
            log.warning("[Bruxos BlockSwap] load device nao e CUDA, pulando.")
            return

        total = len(blocks)
        n_swap = max(0, min(int(blocks_to_swap), total))

        block_sizes = [mm.module_size(b) for b in blocks]
        dm_size = mm.module_size(dm)
        other_size = dm_size - sum(block_sizes)

        # respeita o orcamento de VRAM do comfy: sobe n_swap se o residente nao cabe
        if lowvram_model_memory is not None and lowvram_model_memory < 1e30:
            margin = (max(block_sizes) if block_sizes else 0) + _SWAP_BUFFER_EXTRA

            def resident_bytes(n):
                return other_size + sum(block_sizes[n:])

            while n_swap < total and resident_bytes(n_swap) + margin > lowvram_model_memory:
                n_swap += 1
            if n_swap > blocks_to_swap:
                log.info("[Bruxos BlockSwap] blocks_to_swap {} -> {} pra caber no orcamento "
                         "de {:.0f} MB de VRAM.".format(blocks_to_swap, n_swap,
                                                        lowvram_model_memory / (1024 * 1024)))

        prefix = "diffusion_model.blocks"

        # 1) offload dos blocos de swap primeiro (libera VRAM)
        swapped_bytes = 0
        pinned_bytes = 0
        pinned_ptrs = set()  # dedupe: pesos INT8/quant compartilham storage empacotado
        done = 0
        for i in range(n_swap):
            block = blocks[i]
            try:
                _deactivate_block(block)
                _finalize_tree(patcher, "{}.{}".format(prefix, i), block, offload_device)
                masters = []
                pinned = []
                for t in _iter_block_tensors(block):
                    masters.append((t, t.data))
                    if not pin_masters:
                        continue
                    # pin best-effort: so tensor CPU contiguo e nao ja registrado.
                    # storage duplicado (INT8 packed) -> cudaErrorInvalidValue; aqui
                    # ele e pulado em vez de derrubar o swap inteiro.
                    try:
                        d = t.data
                        if d.device.type != "cpu" or not d.is_contiguous():
                            continue
                        try:
                            ptr = d.untyped_storage().data_ptr()
                        except Exception:
                            ptr = d.data_ptr()
                        if ptr in pinned_ptrs:
                            continue
                        if mm.pin_memory(d):
                            pinned.append(d)
                            pinned_ptrs.add(ptr)
                            pinned_bytes += d.nbytes
                    except Exception:
                        pass  # segue sem pin nesse tensor
                if not hasattr(block, "_bs_orig_forward"):
                    block._bs_orig_forward = block.forward
                    block.forward = _make_swap_forward(block)
                block._bs_state = {
                    "active": True,
                    "load_device": load_device,
                    "masters": masters,
                    "pinned": pinned,
                }
                swapped_bytes += block_sizes[i]
                done += 1
            except Exception as be:
                # rollback: devolve ESTE bloco pra GPU e para de swappar o resto.
                # (melhor menos swap do que blocos presos na CPU -> 400s/it)
                log.warning("[Bruxos BlockSwap] bloco {} falhou no swap ({}); "
                            "mantendo {} swappados, resto residente.".format(i, be, done))
                try:
                    _deactivate_block(block)
                    _finalize_tree(patcher, "{}.{}".format(prefix, i), block,
                                   load_device, unpin_all=True)
                except Exception:
                    pass
                break
        n_swap = done

        # 2) o resto fica 100% residente na GPU (sem caminho de cast por-camada)
        for i in range(n_swap, total):
            block = blocks[i]
            _deactivate_block(block)
            _finalize_tree(patcher, "{}.{}".format(prefix, i), block, load_device, unpin_all=True)

        for sub_name, m in dm.named_modules():
            if sub_name == "blocks" or sub_name.startswith("blocks."):
                continue
            full = "diffusion_model.{}".format(sub_name) if sub_name else "diffusion_model"
            _finalize_module(patcher, full, m, load_device, unpin_all=True)
            if sub_name and "." not in sub_name:
                m.to(load_device)
        for t in list(dm.parameters(recurse=False)) + list(dm.buffers(recurse=False)):
            t.data = t.data.to(load_device)

        resident = dm_size - swapped_bytes
        base.model_lowvram = n_swap > 0
        base.model_loaded_weight_memory = resident
        base.model_offload_buffer_memory = (max(block_sizes) if n_swap > 0 else 0) + _SWAP_BUFFER_EXTRA

        mm.soft_empty_cache()
        log.info("[Bruxos BlockSwap] {}/{} blocos na RAM: {:.0f} MB offloadados "
                 "({:.0f} MB pinned), {:.0f} MB residente na GPU.".format(
                     n_swap, total, swapped_bytes / (1024 * 1024),
                     pinned_bytes / (1024 * 1024), resident / (1024 * 1024)))
        print("[Bruxos BlockSwap] {}/{} blocos p/ RAM (pinned={:.0f}MB), residente {:.0f}MB.".format(
            n_swap, total, pinned_bytes / (1024 * 1024), resident / (1024 * 1024)), flush=True)
    except Exception as e:  # nunca derruba o run — no pior caso, sem swap
        log.warning(f"[Bruxos BlockSwap] falha no ON_LOAD ({e}); restaurando e seguindo sem swap.")
        print(f"[Bruxos BlockSwap] falha no ON_LOAD ({e}); restaurando e seguindo sem swap.", flush=True)
        # RESTORE DE EMERGENCIA: garante que nenhum bloco fique preso na CPU com o
        # wrapper de swap (senao o comfy streama de RAM a cada forward -> ~400s/it).
        try:
            _dm = getattr(patcher.model, "diffusion_model", None)
            _blocks = getattr(_dm, "blocks", None)
            _ld = None
            try:
                _ld = torch.device(device_to if device_to is not None else patcher.load_device)
            except Exception:
                _ld = None
            if _blocks is not None and _ld is not None and mm.is_device_cuda(_ld):
                _pref = "diffusion_model.blocks"
                for _i, _b in enumerate(_blocks):
                    try:
                        _deactivate_block(_b)
                        if getattr(_b, "_bs_state", None) is None:
                            _finalize_tree(patcher, "{}.{}".format(_pref, _i), _b,
                                           _ld, unpin_all=True)
                    except Exception:
                        pass
                mm.soft_empty_cache()
                print("[Bruxos BlockSwap] restore de emergencia: blocos devolvidos "
                      "pra GPU (sem swap, mas sem penalidade de streaming).", flush=True)
        except Exception:
            pass


def _on_detach(patcher, unpatch_all):
    try:
        dm = getattr(patcher.model, "diffusion_model", None)
        blocks = getattr(dm, "blocks", None)
        if blocks is None:
            return
        for block in blocks:
            _deactivate_block(block)
    except Exception:
        pass


class BruxosBlockSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "MODEL do loader (coloque DEPOIS dos LoRA loaders). Ligue o model_high/model_low aqui e a saida no node que usa o modelo (I2V/Sampler)."}),
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 80, "step": 1,
                    "tooltip": "Quantos blocos ficam na RAM e sao streamados pra GPU por forward. Suba se ainda der OOM. E aumentado automaticamente se o residente nao couber na VRAM. Wan 14B tem 40 blocos."}),
                "pin_memory": ("BOOLEAN", {"default": True,
                    "tooltip": "Page-lock das copias na RAM (transferencia PCIe mais rapida). Custa a mesma RAM nao-swappavel. Se der 'lixo' na imagem/erro CUDA em bf16+LoRA, tente lancar com --disable-pinned-memory (ver README do JITBlockSwap)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("O mesmo MODEL com o block swap armado (age no load). Ligue no node que consome o modelo.",)
    FUNCTION = "apply"
    CATEGORY = CAT
    DESCRIPTION = (
        "Block Swap (RAM Offload) pro Wan 2.2 / Bernini-R nativo: streama os primeiros N blocos do "
        "transformer da RAM pra GPU sob demanda -> roda modelos maiores que a VRAM. LoRA e baked no load. "
        "Custo ~0 quando compute-bound (alta resolucao). PRECISA de --disable-dynamic-vram (senao e no-op "
        "e avisa no console). Portado do ComfyUI-JITBlockSwap. Cadeia: Loader -> BlockSwap -> I2V/Sampler."
    )

    def apply(self, model, blocks_to_swap, pin_memory=True):
        if not _OK:
            print("[Bruxos BlockSwap] comfy/torch indisponivel neste build; passando o modelo intacto.", flush=True)
            return (model,)
        if int(blocks_to_swap) <= 0:
            return (model,)
        m = model.clone()
        try:
            m.add_callback_with_key(
                CallbacksMP.ON_LOAD, CALLBACK_KEY,
                functools.partial(_on_load, blocks_to_swap=int(blocks_to_swap),
                                  pin_masters=bool(pin_memory)))
            m.add_callback_with_key(CallbacksMP.ON_DETACH, CALLBACK_KEY, _on_detach)
        except Exception as e:
            print(f"[Bruxos BlockSwap] nao consegui registrar callbacks ({e}); modelo intacto.", flush=True)
            return (model,)
        print(f"[Bruxos BlockSwap] armado: {int(blocks_to_swap)} blocos p/ RAM (pinned={bool(pin_memory)}). "
              f"Age no proximo load. (Precisa de --disable-dynamic-vram; senao vira no-op.)", flush=True)
        return (m,)


NODE_CLASS_MAPPINGS = {"BruxosBlockSwap": BruxosBlockSwap}
NODE_DISPLAY_NAME_MAPPINGS = {"BruxosBlockSwap": "Block Swap RAM Offload (Bruxos)"}
